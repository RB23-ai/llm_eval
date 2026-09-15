"""
APPLICATION-LEVEL EVAL — Quality, Safety, Ops

Treats the RAG chatbot as a finished product rather than a set of
components. Three categories (see doc_07_safety_metrics.txt and the RAG
Operational Evals framework: latency / cost / reliability):

  Quality: correctness against the golden reference answers.
  Safety:  toxicity screen, PII leakage screen, jailbreak-resistance probes.
           (Scope adherence is its own suite: eval_scope_adherence.py.)
  Ops:     latency distributions (P50/P95/P99, split into retrieval vs.
           generation), reliability (success/error rate), and cost
           (input vs. output, plus a projected monthly bill) -- so a change
           that is "more accurate" but 10x slower, flakier, or 10x more
           expensive doesn't ship silently.
"""
from __future__ import annotations

from evals.common import load_golden_dataset, write_suite_results
from evals.metrics import (
    JAILBREAK_PROBES,
    get_judge,
    jailbreak_resisted,
    percentile,
    pii_flag,
    toxicity_flag,
)
from src.config import SETTINGS
from src.factory import build_judge_llm, build_pipeline


def _run_ops_measurements(pipeline, questions: list[str]) -> dict:
    """Runs each question through the pipeline with warm-up + repeated timed
    runs, per the ops-eval practice of ignoring cold starts and averaging
    out API/model noise rather than trusting a single measurement."""
    total_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    input_costs: list[float] = []
    output_costs: list[float] = []
    successes = 0
    errors = 0
    error_detail: list[dict] = []

    for question in questions:
        # Untimed warm-up run(s), discarded, so first-call overhead
        # (model/container init) doesn't pollute the percentile calculation.
        for _ in range(SETTINGS.ops_warmup_runs):
            try:
                pipeline.answer(question)
            except Exception:  # noqa: BLE001 - warm-up failures aren't scored
                pass

        for _ in range(SETTINGS.ops_repeat_runs):
            try:
                result = pipeline.answer(question)
                successes += 1
                total_latencies.append(result.latency_seconds)
                retrieval_latencies.append(result.retrieval_latency_seconds)
                generation_latencies.append(result.generation_latency_seconds)
                input_costs.append(result.input_cost_usd_est)
                output_costs.append(result.output_cost_usd_est)
            except Exception as exc:  # noqa: BLE001 - reliability eval wants the failure, not a crash
                errors += 1
                error_detail.append({"question": question, "error": f"{type(exc).__name__}: {exc}"})

    total_runs = successes + errors
    avg_cost_per_query = (sum(input_costs) + sum(output_costs)) / successes if successes else 0.0

    return {
        "total_runs": total_runs,
        "successes": successes,
        "errors": errors,
        "success_rate": successes / total_runs if total_runs else 0.0,
        "error_detail": error_detail,
        "latency": {
            "total_p50_seconds": percentile(total_latencies, 50),
            "total_p95_seconds": percentile(total_latencies, 95),
            "total_p99_seconds": percentile(total_latencies, 99),
            "retrieval_p95_seconds": percentile(retrieval_latencies, 95),
            "generation_p95_seconds": percentile(generation_latencies, 95),
        },
        "cost": {
            "avg_input_cost_usd_per_query": (sum(input_costs) / successes) if successes else 0.0,
            "avg_output_cost_usd_per_query": (sum(output_costs) / successes) if successes else 0.0,
            "avg_cost_usd_per_query": avg_cost_per_query,
            "projected_daily_cost_usd": avg_cost_per_query * SETTINGS.ops_daily_query_volume,
            "projected_monthly_cost_usd": avg_cost_per_query * SETTINGS.ops_daily_query_volume * 30,
            "assumed_daily_query_volume": SETTINGS.ops_daily_query_volume,
        },
        "total_prompt_tokens_est": None,  # filled in by caller from the single-pass quality loop
    }


def _run_application_suite():
    pipeline = build_pipeline()
    judge = get_judge(SETTINGS.judge_provider, build_judge_llm())
    golden = load_golden_dataset()

    # --- Quality (single pass; ops measurements below handle repeats) -----
    quality_rows = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for item in golden:
        result = pipeline.answer(item["question"])
        correctness = judge.score_correctness(item["question"], result.answer, item["reference_answer"])
        quality_rows.append({"id": item["id"], "correctness": correctness.score})
        total_prompt_tokens += result.prompt_tokens_est
        total_completion_tokens += result.completion_tokens_est

    avg_correctness = sum(r["correctness"] for r in quality_rows) / len(quality_rows)

    # --- Safety --------------------------------------------------------
    safety_rows = []
    for item in golden:
        result = pipeline.answer(item["question"])
        safety_rows.append(
            {
                "id": item["id"],
                "toxicity_flagged": toxicity_flag(result.answer),
                "pii_flags": pii_flag(result.answer),
            }
        )
    toxicity_violations = sum(1 for r in safety_rows if r["toxicity_flagged"])
    pii_violations = sum(1 for r in safety_rows if any(r["pii_flags"].values()))

    jailbreak_rows = []
    for probe in JAILBREAK_PROBES:
        result = pipeline.answer(probe)
        resisted = jailbreak_resisted(result.answer)
        jailbreak_rows.append({"probe": probe, "resisted": resisted, "answer": result.answer})
    jailbreak_failures = sum(1 for r in jailbreak_rows if not r["resisted"])

    # --- Ops: latency distribution, reliability, cost --------------------
    ops = _run_ops_measurements(pipeline, [item["question"] for item in golden])
    ops["llm_provider"] = SETTINGS.llm_provider
    ops["total_prompt_tokens_est"] = total_prompt_tokens
    ops["total_completion_tokens_est"] = total_completion_tokens
    ops["max_p95_latency_gate"] = SETTINGS.max_p95_latency_seconds

    results = {
        "suite": "application_level",
        "quality": {
            "avg_correctness": avg_correctness,
            "per_question": quality_rows,
        },
        "safety": {
            "toxicity_violations": toxicity_violations,
            "pii_violations": pii_violations,
            "jailbreak_failures": jailbreak_failures,
            "jailbreak_probes_tested": len(JAILBREAK_PROBES),
            "per_question": safety_rows,
            "jailbreak_detail": jailbreak_rows,
        },
        "ops": ops,
        "passed_gate": (
            toxicity_violations == 0
            and pii_violations == 0
            and jailbreak_failures == 0
            and ops["success_rate"] == 1.0
            and ops["latency"]["total_p95_seconds"] <= SETTINGS.max_p95_latency_seconds
        ),
    }
    write_suite_results("application_level", results)
    return results


def test_no_toxicity_or_pii_violations():
    results = _run_application_suite()
    assert results["safety"]["toxicity_violations"] == 0
    assert results["safety"]["pii_violations"] == 0


def test_jailbreak_probes_are_resisted():
    results = _run_application_suite()
    assert results["safety"]["jailbreak_failures"] == 0


def test_p95_latency_meets_gate():
    results = _run_application_suite()
    assert results["ops"]["latency"]["total_p95_seconds"] <= SETTINGS.max_p95_latency_seconds


def test_reliability_meets_gate():
    results = _run_application_suite()
    assert results["ops"]["success_rate"] == 1.0, (
        f"{results['ops']['errors']}/{results['ops']['total_runs']} runs failed: "
        f"{results['ops']['error_detail']}"
    )


if __name__ == "__main__":
    import json

    print(json.dumps(_run_application_suite(), indent=2))
