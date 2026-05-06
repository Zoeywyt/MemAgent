from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
MOODBENCH_ROOTS = [
    PROJECT_ROOT / "MoodBench",
    WORKSPACE_ROOT / "MoodBench",
]
# DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"
DEFAULT_BASE_MODEL = "/data/home/wls_cwz/data/model/Qwen/Qwen2.5-7B-Instruct/"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "models" / "output_sft_7b"
DEFAULT_DPO_ADAPTER = PROJECT_ROOT / "models" / "output_dpo_7b"

PQEMOTION_CONFIGS = {
    "PQEmotion1": "test_PQEmotion1.yaml",
    "PQEmotion2": "test_PQEmotion2.yaml",
    "PQEmotion3": "test_PQEmotion3.yaml",
    "PQEmotion4": "test_PQEmotion4.yaml",
    "PQEmotion5": "test_PQEmotion5.yaml",
}


def find_moodbench_root(explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit else MOODBENCH_ROOTS
    for candidate in candidates:
        if candidate and _is_valid_moodbench_root(candidate):
            return candidate.resolve()
    expected = ", ".join(str(path) for path in MOODBENCH_ROOTS)
    raise FileNotFoundError(f"MoodBench root not found. Expected one of: {expected}")


def _is_valid_moodbench_root(path: Path) -> bool:
    return (path / "src" / "PQAEF" / "run.py").exists() and (path / "test").exists()


def run_moodbench(
    *,
    models: Iterable[str],
    datasets: Iterable[str],
    base_model: Path = DEFAULT_BASE_MODEL,
    sft_adapter: Path = DEFAULT_SFT_ADAPTER,
    dpo_adapter: Path = DEFAULT_DPO_ADAPTER,
    moodbench_root: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
    top_p: float = 0.95,
    limit: int | None = None,
    aggregate: bool = True,
) -> None:
    moodbench_root = find_moodbench_root(moodbench_root).resolve()
    pythonpath_root = (moodbench_root / "src").resolve()
    launch_env = os.environ.copy()
    existing_pythonpath = launch_env.get("PYTHONPATH", "")
    launch_env["PYTHONPATH"] = (
        f"{pythonpath_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(pythonpath_root)
    )

    model_paths: dict[str, Path | None] = {
        "base": None,
        "sft": sft_adapter,
        "dpo": dpo_adapter,
    }
    output_dir = (output_dir or (moodbench_root / "output" / "test")).resolve()
    generated_config_dir = PROJECT_ROOT / "evaluate" / "generated_moodbench_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        model_name = model_name.strip().lower()
        if model_name not in model_paths:
            raise ValueError(f"Unsupported MoodBench model: {model_name}")
        adapter_path = model_paths[model_name]
        for dataset in datasets:
            config_path = _write_config(
                dataset=dataset,
                model_name=model_name,
                base_model=base_model,
                adapter_path=adapter_path,
                output_dir=output_dir,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                limit=limit,
                generated_config_dir=generated_config_dir,
                moodbench_root=moodbench_root,
            )
            print(f"[MoodBench] running {model_name}/{dataset}")
            run_script = (moodbench_root / "src" / "PQAEF" / "run.py").resolve()
            subprocess.run(
                [sys.executable, str(run_script), "--config", str(config_path)],
                cwd=str(moodbench_root),
                env=launch_env,
                check=True,
            )

    if aggregate:
        score_script = (moodbench_root / "calculate_weighted_scores.py").resolve()
        subprocess.run(
            [sys.executable, str(score_script), "--results_path", str(output_dir)],
            cwd=str(moodbench_root),
            env=launch_env,
            check=True,
        )


def _write_config(
    *,
    dataset: str,
    model_name: str,
    base_model: Path,
    adapter_path: Path | None,
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    limit: int | None,
    generated_config_dir: Path,
    moodbench_root: Path,
) -> Path:
    if dataset not in PQEMOTION_CONFIGS:
        raise ValueError(f"Unsupported MoodBench dataset: {dataset}")
    source_config = moodbench_root / "test" / PQEMOTION_CONFIGS[dataset]
    with source_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if limit is not None:
        for loader in config.get("data_loaders", {}).values():
            loader["num"] = int(limit)

    # Resolve to absolute paths: the subprocess runs with cwd=moodbench_root, so
    # relative paths in the YAML (e.g. "./models/output_sft_7b") would be looked
    # up under MoodBench/ and miss the real checkpoints under MemAgent/.
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if model_name == "base":
        model_config = {
            "class": "LocalModel",
            "name": model_name,
            "config": {
                "model_path": str(Path(base_model).resolve()),
                "batch_size": batch_size,
                "device_ids": [0],
                "generation_kwargs": generation_kwargs,
            },
        }
    else:
        if adapter_path is None:
            raise ValueError(f"Adapter path is required for MoodBench model: {model_name}")
        model_config = {
            "class": "PeftModel",
            "name": model_name,
            "config": {
                "base_model_path": str(Path(base_model).resolve()),
                "adapter_path": str(Path(adapter_path).resolve()),
                "batch_size": batch_size,
                "device_ids": [0],
                "tokenizer_from_adapter": True,
                "generation_kwargs": generation_kwargs,
            },
        }
    config["models"] = {model_name: model_config}
    for task in config.get("tasks", []):
        task.setdefault("config", {})["llm_model_name"] = model_name
    config.setdefault("data_dumper", {})["output_dir"] = str(output_dir)

    config_path = generated_config_dir / f"{model_name}_{dataset}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return config_path


def copy_moodbench_scores(destination: Path) -> None:
    try:
        source = find_moodbench_root() / "result_analyze" / "scores.json"
    except FileNotFoundError:
        return
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def read_moodbench_scores() -> dict:
    try:
        source = find_moodbench_root() / "result_analyze" / "scores.json"
    except FileNotFoundError:
        return {}
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))
