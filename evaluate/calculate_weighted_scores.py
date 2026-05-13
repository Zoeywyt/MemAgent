# -*- coding: utf-8 -*-
"""
Official MoodBench weighted-score aggregation script, adapted only by path.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional


DEFAULT_OUTPUT_DIR = "result_analyze"
DEFAULT_BASE_RESULT_PATH = "./output/test"
RESULT_FILENAME = "statistical_analysis/result_stats.json"

SUCCESSFULLY_LOADED_DATASETS = set()
BASE_RESULT_PATH = DEFAULT_BASE_RESULT_PATH


def setup_logging(log_file_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, "w", "utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def get_dataset_score(dataset_name: str, model_path: str) -> Optional[float]:
    score_file_path = os.path.join(model_path, dataset_name, RESULT_FILENAME)
    if not os.path.exists(score_file_path):
        logging.warning(
            f"Result file not found: {dataset_name} (model: {os.path.basename(model_path)}). Dataset excluded from calculation"
        )
        return None
    try:
        with open(score_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            first_value = next(iter(data.values()))
            if not isinstance(first_value, (int, float)):
                logging.error(f"First value in {score_file_path} is not numeric, dataset excluded from calculation")
                return None

            score_percentage = float(first_value) * 100
            SUCCESSFULLY_LOADED_DATASETS.add(dataset_name)
            return score_percentage
    except (json.JSONDecodeError, StopIteration, AttributeError) as e:
        logging.error(f"Failed to parse JSON file: {score_file_path}, error: {e}. Dataset excluded from calculation")
        return None


def generate_model_datasets_json(model_path: str) -> Dict[str, float]:
    model_name = os.path.basename(model_path)
    logging.info(f"Processing model: {model_name}")

    all_datasets = set()
    if os.path.exists(model_path):
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isdir(item_path):
                result_file = os.path.join(item_path, RESULT_FILENAME)
                if os.path.exists(result_file):
                    all_datasets.add(item)

    logging.info(f"Model {model_name} found {len(all_datasets)} datasets")

    dataset_scores = {}
    for dataset_name in sorted(all_datasets):
        score = get_dataset_score(dataset_name, model_path)
        if score is not None:
            dataset_scores[dataset_name] = round(score, 2)
        else:
            dataset_scores[dataset_name] = -1

    valid_scores = sum(1 for score in dataset_scores.values() if score != -1)
    invalid_scores = len(dataset_scores) - valid_scores
    logging.info(f"Model {model_name} statistics: {valid_scores} valid scores, {invalid_scores} missing scores")

    return dataset_scores


def generate_all_models_json() -> None:
    logging.info("Starting to traverse all models...")

    if not os.path.exists(BASE_RESULT_PATH):
        logging.error(f"Base result path does not exist: {BASE_RESULT_PATH}")
        return

    model_dirs = []
    for item in os.listdir(BASE_RESULT_PATH):
        item_path = os.path.join(BASE_RESULT_PATH, item)
        if os.path.isdir(item_path):
            model_dirs.append(item_path)

    if not model_dirs:
        logging.error(f"No model directories found in {BASE_RESULT_PATH}")
        return

    logging.info(f"Found {len(model_dirs)} model directories")

    all_models_scores = {}
    for model_path in sorted(model_dirs):
        model_name = os.path.basename(model_path)
        logging.info(f"\n--- Processing model: {model_name} ---")

        global SUCCESSFULLY_LOADED_DATASETS
        SUCCESSFULLY_LOADED_DATASETS = set()

        model_scores = generate_model_datasets_json(model_path)
        valid_scores = {k: v for k, v in model_scores.items() if v != -1}
        if valid_scores:
            all_models_scores[model_name] = valid_scores
            logging.info(f"Model {model_name} added {len(valid_scores)} valid dataset scores")
        else:
            logging.warning(f"Model {model_name} has no valid dataset scores")

    output_file = os.path.join(DEFAULT_OUTPUT_DIR, "scores.json")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_models_scores, f, ensure_ascii=False, indent=4)

    logging.info(f"\nMerged dataset score JSON file saved to: {output_file}")
    logging.info(f"Total processed models: {len(all_models_scores)}")


def main():
    parser = argparse.ArgumentParser(description="Traverse model directories and calculate dataset scores.")
    parser.add_argument(
        "--results_path",
        type=str,
        default=DEFAULT_BASE_RESULT_PATH,
        help=f"Root directory path for evaluation results. Default: {DEFAULT_BASE_RESULT_PATH}",
    )
    args = parser.parse_args()

    global BASE_RESULT_PATH
    BASE_RESULT_PATH = args.results_path

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    setup_logging(os.path.join(DEFAULT_OUTPUT_DIR, "score_calculation.log"))

    logging.info("--- Starting calculation of dataset scores for all models ---")
    logging.info(f"Using result path: {BASE_RESULT_PATH}")
    logging.info("Starting to generate dataset score JSON file for all models...")
    generate_all_models_json()
    logging.info("--- Analysis completed ---")


if __name__ == "__main__":
    main()
