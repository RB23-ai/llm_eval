#!/usr/bin/env python3
"""
Quick manual demo: ask the RAG pipeline a question and see the retrieved
context, the answer, and ops metadata. Not part of the eval suite — just a
convenience script.

    python scripts/ask.py "What does Recall at K measure?"
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from src.factory import build_pipeline  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question here"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    pipeline = build_pipeline()
    result = pipeline.answer(question)

    print(f"Question: {result.question}\n")
    print("Retrieved context:")
    for doc_id, chunk in zip(result.context_doc_ids, result.context_chunks):
        print(f"  [{doc_id}] {chunk[:120]}...")
    print(f"\nAnswer: {result.answer}\n")
    print(
        f"(latency={result.latency_seconds:.3f}s, "
        f"prompt_tokens~{result.prompt_tokens_est}, "
        f"completion_tokens~{result.completion_tokens_est}, "
        f"cost~${result.cost_usd_est:.5f})"
    )


if __name__ == "__main__":
    main()
