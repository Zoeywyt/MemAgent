from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpsycoun_runner import copy_cpsycoun_results, run_cpsycoun
from moodbench_runner import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DPO_ADAPTER,
    DEFAULT_SFT_ADAPTER,
    copy_moodbench_scores,
    read_moodbench_scores,
    run_moodbench,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "evaluate" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoodBench + CPsyCoun evaluation pipeline.")
    parser.add_argument("--stage", choices=["moodbench", "cpsycoun", "all"], default="all")
    parser.add_argument("--models", default="base,sft,dpo")
    parser.add_argument("--datasets", default="PQEmotion1,PQEmotion2,PQEmotion3,PQEmotion4,PQEmotion5")
    parser.add_argument("--base-model", type=Path, default="/data/home/wls_cwz/data/model/Qwen/Qwen2.5-7B-Instruct/")
    parser.add_argument("--sft-adapter", type=Path, default="./models/output_sft_7b")
    parser.add_argument("--dpo-adapter", type=Path, default="./models/output_dpo_7b")
    parser.add_argument("--moodbench-root", type=Path, default="./MoodBench")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None, help="Override dataset size for quick tests.")
    parser.add_argument("--skip-moodbench-aggregate", action="store_true")
    parser.add_argument("--cpsycoun-root", type=Path, default="./CPsyCoun")
    parser.add_argument("--cpsycoun-input", type=Path, default=None)
    parser.add_argument("--cpsycoun-output-dir", type=Path, default=None)
    parser.add_argument("--cpsycoun-use-existing-input", action="store_true")
    parser.add_argument("--cpsycoun-dry-run", action="store_true")
    parser.add_argument("--cpsycoun-max-workers", type=int, default=30)
    parser.add_argument("--cpsycoun-max-history-length", type=int, default=None)
    parser.add_argument("--skip-cpsycoun-sort", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in {"moodbench", "all"}:
        run_moodbench(
            models=[item.strip() for item in args.models.split(",") if item.strip()],
            datasets=[item.strip() for item in args.datasets.split(",") if item.strip()],
            base_model=args.base_model,
            sft_adapter=args.sft_adapter,
            dpo_adapter=args.dpo_adapter,
            moodbench_root=args.moodbench_root,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
            aggregate=not args.skip_moodbench_aggregate,
        )
        copy_moodbench_scores(REPORT_DIR / "moodbench_scores.json")

    if args.stage in {"cpsycoun", "all"}:
        run_cpsycoun(
            cpsycoun_root=args.cpsycoun_root,
            models=[item.strip() for item in args.models.split(",") if item.strip()],
            base_model=args.base_model,
            sft_adapter=args.sft_adapter,
            dpo_adapter=args.dpo_adapter,
            input_file=args.cpsycoun_input,
            output_dir=args.cpsycoun_output_dir,
            generate_inputs=not args.cpsycoun_use_existing_input,
            dry_run=args.cpsycoun_dry_run,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_workers=args.cpsycoun_max_workers,
            max_history_length=args.cpsycoun_max_history_length,
            limit=args.limit,
            sort_output=not args.skip_cpsycoun_sort,
        )
        copy_cpsycoun_results(REPORT_DIR / "cpsycoun_results")

    summary = {
        "moodbench_scores": read_moodbench_scores(),
        "cpsycoun_results_dir": str(REPORT_DIR / "cpsycoun_results"),
    }
    (REPORT_DIR / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
