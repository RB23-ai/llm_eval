#!/usr/bin/env python3
"""
Runs the full eval suite once and saves a flat metric snapshot to disk --
this is "Save Baseline" or "Run Candidate Eval" in the regression-testing
workflow (see README "Regression testing workflow").

    python scripts/capture_metrics.py --out reports/baseline.json
    # ... change some code/config ...
    python scripts/capture_metrics.py --out reports/candidate.json
    python scripts/compare.py
    python scripts/promote.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.eval_application import _run_application_suite  # noqa: E402
from evals.eval_generator import _run_generator_suite  # noqa: E402
from evals.eval_judge_validation import _run_judge_validation_suite  # noqa: E402
from evals.eval_pipeline import _run_pipeline_suite  # noqa: E402
from evals.eval_retriever import _run_retriever_suite  # noqa: E402
from evals.eval_scope_adherence import _run_scope_suite  # noqa: E402
from evals.flatten import flatten_metrics  # noqa: E402
from src.config import SETTINGS  # noqa: E402


def capture(out_path: str) -> dict:
    suite_results = {
        "judge_validation": _run_judge_validation_suite(),
        "component_retriever": _run_retriever_suite(),
        "component_generator": _run_generator_suite(),
        "pipeline_rag_triad": _run_pipeline_suite(),
        "scope_adherence": _run_scope_suite(),
        "application_level": _run_application_suite(),
    }
    metrics = flatten_metrics(suite_results)

    snapshot = {
        "captured_at_unix": time.time(),
        "config": {
            "llm_provider": SETTINGS.llm_provider,
            "embedding_provider": SETTINGS.embedding_provider,
            "judge_provider": SETTINGS.judge_provider,
            "vector_store": SETTINGS.vector_store,
            "top_k": SETTINGS.top_k,
            "chunk_size_words": SETTINGS.chunk_size_words,
        },
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/baseline.json", help="Where to save the metric snapshot")
    args = parser.parse_args()

    print(f"Capturing metrics with config: LLM_PROVIDER={SETTINGS.llm_provider!r}, "
          f"TOP_K={SETTINGS.top_k}, CHUNK_SIZE_WORDS={SETTINGS.chunk_size_words} ...")
    snapshot = capture(args.out)
    print(f"Wrote {len(snapshot['metrics'])} metrics to {args.out}")
    for path, value in sorted(snapshot["metrics"].items()):
        print(f"  {path} = {value}")


if __name__ == "__main__":
    main()
