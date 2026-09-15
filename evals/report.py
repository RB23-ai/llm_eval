"""Builds a single human-readable Markdown report from the JSON files each
eval suite writes to reports/. Used by run_evals.py and by CI to publish a
readable summary alongside the pass/fail exit code."""
from __future__ import annotations

import json
import os

from evals.common import REPORTS_DIR

SUITES = [
    "judge_validation",
    "component_retriever",
    "component_generator",
    "pipeline_rag_triad",
    "scope_adherence",
    "application_level",
]


def _load(suite: str) -> dict | None:
    path = os.path.join(REPORTS_DIR, f"{suite}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_markdown_report() -> str:
    lines = ["# RAG Evaluation Report", ""]
    all_passed = True

    judge = _load("judge_validation")
    if judge:
        if judge["gate_enforced"]:
            gate_line = f"- Gate: {'✅ PASS' if judge['passed_gate'] else '❌ FAIL'} (enforced)"
        else:
            status = "within threshold" if judge["meets_gate"] else "OUTSIDE threshold"
            gate_line = (
                f"- Gate: ⚠️ ADVISORY ONLY, not enforced for judge_provider='mock' "
                f"(currently {status}, see README \"Judge validation\")"
            )
        lines += [
            "## 0. Judge Validation — is the judge trustworthy?",
            f"- Judge provider: `{judge['judge_provider']}` | Examples: {judge['n_examples']}",
            f"- Faithfulness MAE: **{judge['faithfulness_mae']:.2f}**",
            f"- Answer relevance MAE: **{judge['answer_relevance_mae']:.2f}**",
            f"- Overall MAE: **{judge['overall_mae']:.2f}** (trust gate: <= {judge['max_judge_mae_gate']})",
            gate_line,
            "",
        ]
        all_passed &= judge["passed_gate"]

    retriever = _load("component_retriever")
    if retriever:
        lines += [
            "## 1. Component Level — Retriever",
            f"- Recall@{retriever['top_k']}: **{retriever['avg_recall_at_k']:.2f}** "
            f"(gate: >= {retriever['min_recall_gate']})",
            f"- Precision@{retriever['top_k']}: **{retriever['avg_precision_at_k']:.2f}**",
            f"- Embedding provider: `{retriever['embedding_provider']}`",
            f"- Gate: {'✅ PASS' if retriever['passed_gate'] else '❌ FAIL'}",
            "",
        ]
        all_passed &= retriever["passed_gate"]

    generator = _load("component_generator")
    if generator:
        lines += [
            "## 2. Component Level — Generator",
            f"- Faithfulness: **{generator['avg_faithfulness']:.2f}** (gate: >= {generator['min_faithfulness_gate']})",
            f"- Answer relevance: **{generator['avg_answer_relevance']:.2f}** (gate: >= {generator['min_relevance_gate']})",
            f"- Correctness vs reference: **{generator['avg_correctness']:.2f}**",
            f"- LLM provider: `{generator['llm_provider']}` | Judge: `{generator['judge_provider']}`",
            f"- Gate: {'✅ PASS' if generator['passed_gate'] else '❌ FAIL'}",
            "",
        ]
        all_passed &= generator["passed_gate"]

    pipeline = _load("pipeline_rag_triad")
    if pipeline:
        lines += [
            "## 3. Pipeline Level — RAG Triad",
            f"- Context relevance: **{pipeline['avg_context_relevance']:.2f}**",
            f"- Faithfulness: **{pipeline['avg_faithfulness']:.2f}** (gate: >= {pipeline['min_faithfulness_gate']})",
            f"- Answer relevance: **{pipeline['avg_answer_relevance']:.2f}** (gate: >= {pipeline['min_relevance_gate']})",
            f"- Gate: {'✅ PASS' if pipeline['passed_gate'] else '❌ FAIL'}",
            "",
        ]
        all_passed &= pipeline["passed_gate"]

    scope = _load("scope_adherence")
    if scope:
        cat_summary = ", ".join(f"{cat}={rate:.2f}" for cat, rate in scope["category_pass_rates"].items())
        lines += [
            "## 4. Safety — Scope Adherence",
            f"- Overall pass rate: **{scope['pass_rate']:.2f}** (gate: >= {scope['min_scope_adherence_gate']})",
            f"- By category: {cat_summary}",
            f"- Gate: {'✅ PASS' if scope['passed_gate'] else '❌ FAIL'}",
            "",
        ]
        all_passed &= scope["passed_gate"]

    app = _load("application_level")
    if app:
        ops = app["ops"]
        lines += [
            "## 5. Application Level — Quality / Safety / Ops",
            f"- Correctness: **{app['quality']['avg_correctness']:.2f}**",
            f"- Toxicity violations: **{app['safety']['toxicity_violations']}**",
            f"- PII violations: **{app['safety']['pii_violations']}**",
            f"- Jailbreak probes resisted: **{app['safety']['jailbreak_probes_tested'] - app['safety']['jailbreak_failures']}"
            f"/{app['safety']['jailbreak_probes_tested']}**",
            f"- Reliability: **{ops['success_rate']:.1%}** success ({ops['successes']}/{ops['total_runs']} runs)",
            f"- Latency — P50: **{ops['latency']['total_p50_seconds']*1000:.1f}ms**, "
            f"P95: **{ops['latency']['total_p95_seconds']*1000:.1f}ms** (gate: <= {ops['max_p95_latency_gate']}s), "
            f"P99: **{ops['latency']['total_p99_seconds']*1000:.1f}ms**",
            f"- Latency breakdown (P95) — retrieval: **{ops['latency']['retrieval_p95_seconds']*1000:.1f}ms**, "
            f"generation: **{ops['latency']['generation_p95_seconds']*1000:.1f}ms**",
            f"- Cost — avg/query: **${ops['cost']['avg_cost_usd_per_query']:.6f}** "
            f"(input ${ops['cost']['avg_input_cost_usd_per_query']:.6f} + "
            f"output ${ops['cost']['avg_output_cost_usd_per_query']:.6f}), "
            f"projected monthly at {ops['cost']['assumed_daily_query_volume']:,} queries/day: "
            f"**${ops['cost']['projected_monthly_cost_usd']:,.2f}** (provider: `{ops['llm_provider']}`)",
            f"- Gate: {'✅ PASS' if app['passed_gate'] else '❌ FAIL'}",
            "",
        ]
        all_passed &= app["passed_gate"]

    lines += ["## Overall", f"**{'✅ ALL GATES PASSED' if all_passed else '❌ ONE OR MORE GATES FAILED'}**", ""]
    return "\n".join(lines), all_passed


def write_report() -> bool:
    markdown, all_passed = build_markdown_report()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "eval_report.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    return all_passed
