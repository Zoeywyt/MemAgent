from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .cpsycoun_evaluator import run_cpsycoun_evaluation
    from .model_runner import DryRunRunner, LocalBaseRunner, LocalLoRARunner
except ImportError:
    from cpsycoun_evaluator import run_cpsycoun_evaluation
    from model_runner import DryRunRunner, LocalBaseRunner, LocalLoRARunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
EVALUATE_ROOT = PROJECT_ROOT / "evaluate"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "models" / "output_sft_7b"
DEFAULT_DPO_ADAPTER = PROJECT_ROOT / "models" / "output_dpo_7b"
DEFAULT_CPSYCOUN_ROOTS = [
    PROJECT_ROOT / "CPsyCoun",
    WORKSPACE_ROOT / "CPsyCoun",
]

CLIENT_PREFIXES = ("求助者：", "来访者：", "用户：")
ALL_ROLE_PREFIXES = (
    "求助者：",
    "来访者：",
    "用户：",
    "支持者：",
    "心理咨询师：",
    "咨询师：",
)


def find_cpsycoun_root(explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit else DEFAULT_CPSYCOUN_ROOTS
    for candidate in candidates:
        if not candidate:
            continue
        if candidate.name == "CPsyCounE" and candidate.exists():
            return candidate.parent.resolve()
        if (candidate / "CPsyCounE").exists():
            return candidate.resolve()
    expected = ", ".join(str(path) for path in DEFAULT_CPSYCOUN_ROOTS)
    raise FileNotFoundError(f"CPsyCounE data directory was not found. Expected CPsyCounE in one of: {expected}")


def run_cpsycoun(
    *,
    cpsycoun_root: Path | None = None,
    models: list[str] | None = None,
    base_model: Path = DEFAULT_BASE_MODEL,
    sft_adapter: Path = DEFAULT_SFT_ADAPTER,
    dpo_adapter: Path = DEFAULT_DPO_ADAPTER,
    input_file: Path | None = None,
    output_dir: Path | None = None,
    generate_inputs: bool = True,
    dry_run: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_workers: int = 30,
    max_history_length: int | None = None,
    limit: int | None = None,
    sort_output: bool = True,
) -> None:
    root = find_cpsycoun_root(cpsycoun_root).resolve()
    output_dir = output_dir or (EVALUATE_ROOT / "output" / "cpsycoun")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [item.strip().lower() for item in (models or ["base", "sft", "dpo"]) if item.strip()]
    if not model_names:
        raise ValueError("At least one CPsyCoun target model must be selected.")

    if generate_inputs:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_paths = generate_cpsycoun_inputs(
            root=root,
            output_dir=output_dir / "generated_inputs" / run_id,
            models=model_names,
            base_model=base_model,
            sft_adapter=sft_adapter,
            dpo_adapter=dpo_adapter,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            limit=limit,
            dry_run=dry_run,
        )
    else:
        if input_file is None:
            raise ValueError("--cpsycoun-input is required when --cpsycoun-use-existing-input is set.")
        if not input_file.exists():
            raise FileNotFoundError(f"CPsyCoun input file not found: {input_file}")
        input_paths = {"reference": input_file}

    for model_name, generated_input_path in input_paths.items():
        model_output_dir = output_dir / model_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CPsyCoun] scoring {model_name}: {generated_input_path}")
        run_cpsycoun_evaluation(
            input_file=generated_input_path,
            output_csv=model_output_dir / "evaluation_gpt54.csv",
            max_workers=max_workers,
            max_history_length=max_history_length,
            judge_model="gpt-5.4",
            sort_output=sort_output,
        )


def generate_cpsycoun_inputs(
    *,
    root: Path,
    output_dir: Path,
    models: list[str],
    base_model: Path,
    sft_adapter: Path,
    dpo_adapter: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    limit: int | None,
    dry_run: bool,
) -> dict[str, Path]:
    cases = load_cpsycoun_eval_cases(root / "CPsyCounE", limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    supported_models = {"base", "sft", "dpo"}
    adapters = {"sft": sft_adapter, "dpo": dpo_adapter}
    generated: dict[str, Path] = {}

    for model_name in models:
        if model_name not in supported_models:
            raise ValueError(f"Unsupported CPsyCoun model: {model_name}")
        if dry_run:
            runner = DryRunRunner(model_name)
        elif model_name == "base":
            runner = LocalBaseRunner(
                base_model=base_model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            runner = LocalLoRARunner(
                base_model=base_model,
                adapter_path=adapters[model_name],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        samples = [generate_one_cpsycoun_dialogue(case, runner) for case in cases]
        output_path = output_dir / f"{model_name}_cpsycoun_generated.json"
        output_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        generated[model_name] = output_path
    return generated


def load_cpsycoun_eval_cases(cpsycoun_e_dir: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not cpsycoun_e_dir.exists():
        raise FileNotFoundError(f"CPsyCounE directory not found: {cpsycoun_e_dir}")
    cases: list[dict[str, Any]] = []
    for file_path in sorted(cpsycoun_e_dir.rglob("*.json")):
        topic = file_path.parent.name
        raw_turns = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw_turns, list):
            continue
        client_turns = [
            strip_role_prefix(turn)
            for turn in raw_turns
            if isinstance(turn, str) and turn.strip().startswith(CLIENT_PREFIXES)
        ]
        if client_turns:
            cases.append({"id": f"{topic}_{file_path.stem}", "topic": topic, "client_turns": client_turns})
        if limit is not None and len(cases) >= limit:
            break
    return cases


def strip_role_prefix(text: str) -> str:
    for prefix in ALL_ROLE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text.strip()


def generate_one_cpsycoun_dialogue(case: dict[str, Any], runner: object) -> dict[str, Any]:
    dialogue: list[dict[str, str]] = []
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名中文心理咨询支持者。请在安全、尊重、共情的前提下回应来访者，"
                "结合对话历史，避免泄露隐私，不做武断诊断。"
            ),
        }
    ]

    for client_text in case["client_turns"]:
        dialogue.append({"role": "client", "content": client_text})
        messages.append({"role": "user", "content": client_text})
        counselor_reply = runner.generate(messages)  # type: ignore[attr-defined]
        dialogue.append({"role": "counselor", "content": counselor_reply})
        messages.append({"role": "assistant", "content": counselor_reply})
    return {"id": case["id"], "topic": case["topic"], "dialogue": dialogue}


def copy_cpsycoun_results(destination: Path) -> None:
    source_dir = EVALUATE_ROOT / "output" / "cpsycoun"
    if not source_dir.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source in source_dir.rglob("evaluation_*.csv"):
        relative = source.relative_to(source_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
