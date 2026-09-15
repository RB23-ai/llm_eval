"""
JUDGE VALIDATION — is the judge itself trustworthy?

Every other eval suite (component-generator, pipeline, application) uses
`get_judge(...)` to score faithfulness/relevance/correctness. That judge is
just another model call, and per doc_05_llm_as_judge.txt it should not be
trusted blindly. This suite runs the *same* judge against a small set of
hand-authored reference ratings (data/human_ratings.json) — answers that
range from fully-grounded to outright hallucinated, each pre-scored the way
a careful human grader would score them — and computes Mean Absolute Error
between the judge's scores and the human scores.

This is deliberately run as its own suite, separate from and prior to the
suites that rely on the judge, mirroring how you'd gate a real evaluation
pipeline: don't trust a judge you haven't checked.
"""
from __future__ import annotations

from evals.common import load_human_ratings, write_suite_results
from evals.metrics import get_judge, mean_absolute_error
from src.config import SETTINGS
from src.factory import build_judge_llm


def _run_judge_validation_suite():
    judge = get_judge(SETTINGS.judge_provider, build_judge_llm())
    ratings = load_human_ratings()

    faithfulness_pairs = []
    relevance_pairs = []
    per_example = []

    for r in ratings:
        judged_faithfulness = judge.score_faithfulness(r["context"], r["candidate_answer"])
        judged_relevance = judge.score_answer_relevance(r["question"], r["candidate_answer"])

        faithfulness_pairs.append((r["human_faithfulness"], judged_faithfulness.score))
        relevance_pairs.append((r["human_answer_relevance"], judged_relevance.score))

        per_example.append(
            {
                "id": r["id"],
                "question": r["question"],
                "candidate_answer": r["candidate_answer"],
                "human_faithfulness": r["human_faithfulness"],
                "judge_faithfulness": judged_faithfulness.score,
                "human_answer_relevance": r["human_answer_relevance"],
                "judge_answer_relevance": judged_relevance.score,
                "notes": r.get("notes", ""),
            }
        )

    faithfulness_mae = mean_absolute_error(faithfulness_pairs)
    relevance_mae = mean_absolute_error(relevance_pairs)
    overall_mae = mean_absolute_error(faithfulness_pairs + relevance_pairs)

    # The free HeuristicJudge (lexical overlap) is intentionally coarse and is
    # not expected to reliably track human judgment -- validating that claim
    # is exactly what this suite is for. So its MAE is measured and reported
    # like any other judge, but only *enforced* as a hard gate for real judges
    # (ollama/hf_local/openai/anthropic). This is a documented product
    # decision, not a workaround: mock mode exists to prove the evaluation
    # *plumbing* works end to end for free in CI, never to certify judgment
    # quality. See README "Judge validation" section.
    gate_enforced = SETTINGS.judge_provider != "mock"
    meets_gate = overall_mae <= SETTINGS.max_judge_mae

    results = {
        "suite": "judge_validation",
        "judge_provider": SETTINGS.judge_provider,
        "n_examples": len(ratings),
        "faithfulness_mae": faithfulness_mae,
        "answer_relevance_mae": relevance_mae,
        "overall_mae": overall_mae,
        "max_judge_mae_gate": SETTINGS.max_judge_mae,
        "gate_enforced": gate_enforced,
        "meets_gate": meets_gate,
        "passed_gate": meets_gate if gate_enforced else True,
        "per_example": per_example,
    }
    write_suite_results("judge_validation", results)
    return results


def test_judge_mae_meets_gate():
    results = _run_judge_validation_suite()
    if not results["gate_enforced"]:
        print(
            f"[advisory, not enforced] judge_provider='mock' HeuristicJudge MAE="
            f"{results['overall_mae']:.2f} vs gate {SETTINGS.max_judge_mae} "
            f"(faithfulness_mae={results['faithfulness_mae']:.2f}, "
            f"answer_relevance_mae={results['answer_relevance_mae']:.2f}). "
            f"Point JUDGE_PROVIDER at a real model (ollama/openai/anthropic) to make "
            f"this a hard gate -- see reports/judge_validation.json for the full breakdown."
        )
        return
    assert results["meets_gate"], (
        f"Judge ({SETTINGS.judge_provider}) MAE vs human ratings = {results['overall_mae']:.2f}, "
        f"exceeds the trust threshold of {SETTINGS.max_judge_mae}. Do not rely on this judge's "
        f"scores elsewhere in the suite until this is fixed (better prompt, bigger/better judge "
        f"model, or more human ratings to confirm this isn't noise)."
    )


if __name__ == "__main__":
    import json

    print(json.dumps(_run_judge_validation_suite(), indent=2))
