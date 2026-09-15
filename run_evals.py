#!/usr/bin/env python3
"""
Master control file — the single command a human or a CI pipeline runs.

    python run_evals.py

It runs every suite in evals/ (component -> pipeline -> application level),
writes reports/eval_report.md, and exits with a non-zero status code if any
regression gate defined in src/config.py fails. That non-zero exit code is
what CI (see .github/workflows/ci.yml) uses to block a deployment.
"""
from __future__ import annotations

import sys

from evals.eval_application import _run_application_suite
from evals.eval_generator import _run_generator_suite
from evals.eval_judge_validation import _run_judge_validation_suite
from evals.eval_pipeline import _run_pipeline_suite
from evals.eval_retriever import _run_retriever_suite
from evals.eval_scope_adherence import _run_scope_suite
from evals.report import write_report
from src.config import SETTINGS


def main() -> int:
    print(f"Running eval suite with LLM_PROVIDER={SETTINGS.llm_provider!r}, "
          f"EMBEDDING_PROVIDER={SETTINGS.embedding_provider!r}, "
          f"JUDGE_PROVIDER={SETTINGS.judge_provider!r}, "
          f"VECTOR_STORE={SETTINGS.vector_store!r}\n")

    print("[1/6] Judge validation: is the configured judge trustworthy? ...")
    _run_judge_validation_suite()

    print("[2/6] Component level: retriever ...")
    _run_retriever_suite()

    print("[3/6] Component level: generator ...")
    _run_generator_suite()

    print("[4/6] Pipeline level: RAG triad ...")
    _run_pipeline_suite()

    print("[5/6] Safety: scope adherence ...")
    _run_scope_suite()

    print("[6/6] Application level: quality / safety / ops ...")
    _run_application_suite()

    print("\nWriting reports/eval_report.md ...")
    all_passed = write_report()

    with open("reports/eval_report.md", "r", encoding="utf-8") as f:
        print("\n" + f.read())

    if not all_passed:
        print("REGRESSION GATE FAILED — see reports/eval_report.md", file=sys.stderr)
        return 1

    print("All regression gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
