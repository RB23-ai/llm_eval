"""Shared helpers used by every eval_*.py script."""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DATASET_PATH = os.path.join(ROOT, "data", "golden_dataset.json")
HUMAN_RATINGS_PATH = os.path.join(ROOT, "data", "human_ratings.json")
SCOPE_TEST_CASES_PATH = os.path.join(ROOT, "data", "scope_test_cases.json")
REPORTS_DIR = os.path.join(ROOT, "reports")


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_human_ratings() -> list[dict]:
    """Skips the leading '_note' documentation entry (see the file itself)."""
    with open(HUMAN_RATINGS_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [r for r in rows if "id" in r]


def load_scope_test_cases() -> list[dict]:
    with open(SCOPE_TEST_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_ALL_SUITE_NAMES = [
    "judge_validation",
    "component_retriever",
    "component_generator",
    "pipeline_rag_triad",
    "scope_adherence",
    "application_level",
]


def load_all_suite_results() -> dict[str, dict]:
    """Reads back every per-suite reports/<suite>.json file written by
    write_suite_results(), keyed by suite name. Suites that haven't been run
    yet (no file present) are simply omitted."""
    results = {}
    for suite in _ALL_SUITE_NAMES:
        path = os.path.join(REPORTS_DIR, f"{suite}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                results[suite] = json.load(f)
    return results


def write_suite_results(suite_name: str, results: dict) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{suite_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
