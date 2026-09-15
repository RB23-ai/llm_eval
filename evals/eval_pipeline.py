"""
PIPELINE-LEVEL EVAL — The RAG Triad

Once the retriever and generator are wired together, we evaluate the whole
pipeline using the RAG Triad (see doc_04_rag_triad.txt): the relationship
between the user's Question, the actually-Fetched context, and the
Generated answer. This is where retrieval mistakes and generation mistakes
both show up, since nothing here is hand-fed like in the component evals.
"""
from __future__ import annotations

from evals.common import load_golden_dataset, write_suite_results
from evals.metrics import get_judge
from src.config import SETTINGS
from src.factory import build_judge_llm, build_pipeline


def _run_pipeline_suite():
    pipeline = build_pipeline()
    judge = get_judge(SETTINGS.judge_provider, build_judge_llm())
    golden = load_golden_dataset()

    per_question = []
    for item in golden:
        result = pipeline.answer(item["question"])
        context = "\n\n".join(result.context_chunks)

        context_relevance = judge.score_context_relevance(item["question"], context)
        faithfulness = judge.score_faithfulness(context, result.answer)
        answer_relevance = judge.score_answer_relevance(item["question"], result.answer)

        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "retrieved_doc_ids": result.context_doc_ids,
                "context_relevance": context_relevance.score,
                "faithfulness": faithfulness.score,
                "answer_relevance": answer_relevance.score,
                "latency_seconds": result.latency_seconds,
            }
        )

    def avg(key):
        return sum(r[key] for r in per_question) / len(per_question)

    results = {
        "suite": "pipeline_rag_triad",
        "avg_context_relevance": avg("context_relevance"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_answer_relevance": avg("answer_relevance"),
        "min_faithfulness_gate": SETTINGS.min_faithfulness,
        "min_relevance_gate": SETTINGS.min_answer_relevance,
        "passed_gate": avg("faithfulness") >= SETTINGS.min_faithfulness
        and avg("answer_relevance") >= SETTINGS.min_answer_relevance,
        "per_question": per_question,
    }
    write_suite_results("pipeline_rag_triad", results)
    return results


def test_rag_triad_faithfulness_meets_gate():
    results = _run_pipeline_suite()
    assert results["avg_faithfulness"] >= SETTINGS.min_faithfulness


def test_rag_triad_answer_relevance_meets_gate():
    results = _run_pipeline_suite()
    assert results["avg_answer_relevance"] >= SETTINGS.min_answer_relevance


if __name__ == "__main__":
    import json

    print(json.dumps(_run_pipeline_suite(), indent=2))
