"""
COMPONENT-LEVEL EVAL — Generator

Tests the generator in isolation, *before* it is connected to the retriever.
We feed it the golden-dataset question plus the actual source document text
(the correct context, read straight from disk) so any failure here is
attributable to the generator, not to a retrieval mistake. Scored on
faithfulness (grounded in context?) and answer relevance (does it address the
question?) using whichever judge is configured (see evals/metrics.py).
"""
from __future__ import annotations

import os

from evals.common import load_golden_dataset, write_suite_results
from evals.metrics import get_judge
from src.config import SETTINGS
from src.factory import DOCUMENTS_DIR, build_judge_llm, build_pipeline


def _load_doc_text(doc_id: str) -> str:
    with open(os.path.join(DOCUMENTS_DIR, f"{doc_id}.txt"), "r", encoding="utf-8") as f:
        return f.read()


def _run_generator_suite():
    pipeline = build_pipeline()
    judge = get_judge(SETTINGS.judge_provider, build_judge_llm())
    golden = load_golden_dataset()

    per_question = []
    for item in golden:
        context = "\n\n".join(_load_doc_text(doc_id) for doc_id in item["relevant_doc_ids"])
        answer = pipeline.generator.generate(item["question"], [context])

        faithfulness = judge.score_faithfulness(context, answer)
        relevance = judge.score_answer_relevance(item["question"], answer)
        correctness = judge.score_correctness(item["question"], answer, item["reference_answer"])

        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "faithfulness": faithfulness.score,
                "answer_relevance": relevance.score,
                "correctness": correctness.score,
            }
        )

    def avg(key):
        return sum(r[key] for r in per_question) / len(per_question)

    results = {
        "suite": "component_generator",
        "judge_provider": SETTINGS.judge_provider,
        "llm_provider": SETTINGS.llm_provider,
        "avg_faithfulness": avg("faithfulness"),
        "avg_answer_relevance": avg("answer_relevance"),
        "avg_correctness": avg("correctness"),
        "min_faithfulness_gate": SETTINGS.min_faithfulness,
        "min_relevance_gate": SETTINGS.min_answer_relevance,
        "passed_gate": avg("faithfulness") >= SETTINGS.min_faithfulness
        and avg("answer_relevance") >= SETTINGS.min_answer_relevance,
        "per_question": per_question,
    }
    write_suite_results("component_generator", results)
    return results


def test_generator_faithfulness_meets_gate():
    results = _run_generator_suite()
    assert results["avg_faithfulness"] >= SETTINGS.min_faithfulness, (
        f"Generator faithfulness = {results['avg_faithfulness']:.2f} "
        f"is below the required gate of {SETTINGS.min_faithfulness}"
    )


def test_generator_answer_relevance_meets_gate():
    results = _run_generator_suite()
    assert results["avg_answer_relevance"] >= SETTINGS.min_answer_relevance, (
        f"Generator answer relevance = {results['avg_answer_relevance']:.2f} "
        f"is below the required gate of {SETTINGS.min_answer_relevance}"
    )


if __name__ == "__main__":
    import json

    print(json.dumps(_run_generator_suite(), indent=2))
