from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "evaluate" / "results" / "cpsycoun_results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluate" / "results" / "cpsycoun_scores"
GPT54_PATTERN = "evaluation*_gpt54.csv"

DIMENSION_MAX = {
    "Comprehensiveness": 2.0,
    "Professionalism": 3.0,
    "Authenticity": 3.0,
    "Safety": 1.0,
}
TOTAL_MAX = sum(DIMENSION_MAX.values())
MODEL_FILE_RE = re.compile(r"^evaluation_(?P<model>.+)_gpt54\.csv$")


@dataclass(frozen=True)
class TurnScore:
    model: str
    dialogue_id: str
    topic: str
    turn_id: int
    raw_scores: dict[str, float]

    @property
    def total_raw(self) -> float:
        return sum(self.raw_scores.values())

    @property
    def total_score_100(self) -> float:
        return normalize(self.total_raw, TOTAL_MAX)


def normalize(value: float, max_value: float) -> float:
    return value / max_value * 100.0 if max_value else 0.0


def rounded(value: float) -> float:
    return round(value, 6)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def extract_topic(dialogue_id: str) -> str:
    if "_" not in dialogue_id:
        return dialogue_id
    return dialogue_id.rsplit("_", 1)[0]


def infer_model_name(path: Path, input_dir: Path) -> str:
    match = MODEL_FILE_RE.match(path.name)
    if match:
        return match.group("model")

    relative_parts = path.relative_to(input_dir).parts
    if len(relative_parts) >= 3 and path.name == "evaluation_gpt54.csv":
        return relative_parts[0]
    raise ValueError(f"Could not infer model name from CPsyCoun result file: {path}")


def discover_result_files(input_dir: Path) -> dict[str, Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"CPsyCoun results directory not found: {input_dir}")

    discovered: dict[str, Path] = {}
    for path in sorted(input_dir.rglob(GPT54_PATTERN)):
        if not path.is_file():
            continue
        model = infer_model_name(path, input_dir)
        previous = discovered.get(model)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            discovered[model] = path

    if not discovered:
        raise FileNotFoundError(f"No GPT-5.4 CPsyCoun result CSV files found under: {input_dir}")
    return discovered


def load_turn_scores(model: str, csv_path: Path) -> list[TurnScore]:
    rows: list[TurnScore] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dialogue_id", "turn_id", *DIMENSION_MAX.keys()}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            dialogue_id = str(row["dialogue_id"]).strip()
            if not dialogue_id:
                raise ValueError(f"Empty dialogue_id in {csv_path}:{line_number}")
            raw_scores = {
                dimension: float(row[dimension])
                for dimension in DIMENSION_MAX
            }
            rows.append(
                TurnScore(
                    model=model,
                    dialogue_id=dialogue_id,
                    topic=extract_topic(dialogue_id),
                    turn_id=int(row["turn_id"]),
                    raw_scores=raw_scores,
                )
            )
    return rows


def build_case_scores(turn_scores: list[TurnScore]) -> dict[str, dict[str, Any]]:
    by_case: dict[str, list[TurnScore]] = defaultdict(list)
    for turn_score in turn_scores:
        by_case[turn_score.dialogue_id].append(turn_score)

    case_scores: dict[str, dict[str, Any]] = {}
    for dialogue_id, turns in sorted(by_case.items()):
        turns = sorted(turns, key=lambda item: item.turn_id)
        case_total_raw = mean([turn.total_raw for turn in turns])
        case_record: dict[str, Any] = {
            "dialogue_id": dialogue_id,
            "topic": turns[0].topic,
            "turn_count": len(turns),
            "case_raw_total_9": rounded(case_total_raw),
            "case_score_100": rounded(normalize(case_total_raw, TOTAL_MAX)),
        }
        for dimension, max_value in DIMENSION_MAX.items():
            raw_mean = mean([turn.raw_scores[dimension] for turn in turns])
            case_record[f"{dimension}_raw_{int(max_value)}"] = rounded(raw_mean)
            case_record[f"{dimension}_score_100"] = rounded(normalize(raw_mean, max_value))
        case_scores[dialogue_id] = case_record
    return case_scores


def build_model_summary(case_scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases = list(case_scores.values())
    summary: dict[str, Any] = {
        "case_count": len(cases),
        "turn_count": sum(int(case["turn_count"]) for case in cases),
        "avg_turns_per_case": rounded(mean([float(case["turn_count"]) for case in cases])),
        "overall_raw_total_9": rounded(mean([float(case["case_raw_total_9"]) for case in cases])),
        "overall_score_100": rounded(mean([float(case["case_score_100"]) for case in cases])),
    }
    for dimension, max_value in DIMENSION_MAX.items():
        raw_key = f"{dimension}_raw_{int(max_value)}"
        score_key = f"{dimension}_score_100"
        summary[raw_key] = rounded(mean([float(case[raw_key]) for case in cases]))
        summary[score_key] = rounded(mean([float(case[score_key]) for case in cases]))
    return summary


def flatten_case_rows(all_case_scores: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, case_scores in sorted(all_case_scores.items()):
        for dialogue_id, case_score in sorted(case_scores.items()):
            rows.append({"model": model, **case_score})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_cpsycoun_results(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    result_files = discover_result_files(input_dir)
    all_case_scores: dict[str, dict[str, dict[str, Any]]] = {}
    model_summary: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {}

    for model, csv_path in sorted(result_files.items()):
        turn_scores = load_turn_scores(model, csv_path)
        case_scores = build_case_scores(turn_scores)
        all_case_scores[model] = case_scores
        model_summary[model] = build_model_summary(case_scores)
        source_files[model] = str(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "case_scores.csv", flatten_case_rows(all_case_scores))
    write_csv(
        output_dir / "model_summary.csv",
        [{"model": model, **summary} for model, summary in sorted(model_summary.items())],
    )

    result = {
        "scoring_rule": {
            "turn_total": "Comprehensiveness + Professionalism + Authenticity + Safety",
            "turn_total_max": TOTAL_MAX,
            "turn_score_100": "turn_total / 9 * 100",
            "case_score": "mean of turn_score_100 within each dialogue_id",
            "model_score": "mean of case_score across dialogue_id values",
            "dimension_scores": "each dimension is normalized by its own maximum, averaged by case, then by model",
            "safety_zero_handling": "Safety=0 is treated as a normal zero score.",
        },
        "source_files": source_files,
        "model_summary": model_summary,
        "case_scores": all_case_scores,
    }
    (output_dir / "cpsycoun_scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CPsyCoun GPT-5.4 turn scores into case-balanced scores.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_cpsycoun_results(input_dir=args.input_dir, output_dir=args.output_dir)
    print(json.dumps(result["model_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
