"""
COMPONENT-LEVEL EVAL — Retriever

Tests the retriever in isolation, before it's wired to the generator. For
every golden-dataset question we know which source document should be
retrieved; we check whether the retriever actually surfaces it in its top K
results using Recall@K and Precision@K (see doc_03_retriever_metrics.txt).
"""
from __future__ import annotations

from evals.common import load_golden_dataset, write_suite_results
from evals.metrics import precision_at_k, recall_at_k
from src.config import SETTINGS
from src.factory import build_pipeline


def _run_retriever_suite():
    pipeline = build_pipeline()
    golden = load_golden_dataset()

    per_question = []
    for item in golden:
        retrieved = pipeline.retriever.retrieve(item["question"], top_k=SETTINGS.top_k)
        retrieved_doc_ids = [r.chunk.doc_id for r in retrieved]
        recall = recall_at_k(retrieved_doc_ids, item["relevant_doc_ids"])
        precision = precision_at_k(retrieved_doc_ids, item["relevant_doc_ids"])
        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieved_doc_ids": retrieved_doc_ids,
                "relevant_doc_ids": item["relevant_doc_ids"],
                "recall_at_k": recall,
                "precision_at_k": precision,
            }
        )

    avg_recall = sum(r["recall_at_k"] for r in per_question) / len(per_question)
    avg_precision = sum(r["precision_at_k"] for r in per_question) / len(per_question)

    results = {
        "suite": "component_retriever",
        "top_k": SETTINGS.top_k,
        "embedding_provider": SETTINGS.embedding_provider,
        "avg_recall_at_k": avg_recall,
        "avg_precision_at_k": avg_precision,
        "min_recall_gate": SETTINGS.min_retriever_recall_at_k,
        "passed_gate": avg_recall >= SETTINGS.min_retriever_recall_at_k,
        "per_question": per_question,
    }
    write_suite_results("component_retriever", results)
    return results


def test_retriever_recall_meets_gate():
    results = _run_retriever_suite()
    assert results["avg_recall_at_k"] >= SETTINGS.min_retriever_recall_at_k, (
        f"Retriever recall@{SETTINGS.top_k} = {results['avg_recall_at_k']:.2f} "
        f"is below the required gate of {SETTINGS.min_retriever_recall_at_k}"
    )


if __name__ == "__main__":
    import json

    print(json.dumps(_run_retriever_suite(), indent=2))
