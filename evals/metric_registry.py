"""
The Metric Registry — one place declaring, for every metric this project
tracks across runs:

  path      : dotted path into the suite result dict (see evals/flatten.py)
  direction : "higher_better" or "lower_better" -- metrics don't all improve
              in the same direction (faithfulness up is good, latency up is
              bad), so compare.py needs this to know what a "regression"
              even means for a given metric.
  tier      : "critical" | "important" | "minor" -- used by scripts/promote.py
              to decide whether a regressed metric should BLOCK a deploy,
              send it to human REVIEW, or just be noted.

This is intentionally a plain data structure, not scattered as assumptions
inside compare.py/promote.py, so extending the eval suite with a new metric
means adding one line here rather than hunting through scripts for hardcoded
logic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    path: str
    direction: str  # "higher_better" | "lower_better"
    tier: str  # "critical" | "important" | "minor"
    description: str


METRIC_REGISTRY: list[MetricSpec] = [
    # --- Judge validation --------------------------------------------------
    MetricSpec("judge_validation.overall_mae", "lower_better", "important",
               "How far the judge's scores drift from human ratings"),
    MetricSpec("judge_validation.faithfulness_mae", "lower_better", "important",
               "Judge accuracy specifically on faithfulness scoring"),
    MetricSpec("judge_validation.answer_relevance_mae", "lower_better", "important",
               "Judge accuracy specifically on answer-relevance scoring"),

    # --- Component: retriever ------------------------------------------
    MetricSpec("component_retriever.avg_recall_at_k", "higher_better", "critical",
               "Fraction of relevant documents the retriever surfaces in top K"),
    MetricSpec("component_retriever.avg_precision_at_k", "higher_better", "minor",
               "Fraction of retrieved documents that are actually relevant"),

    # --- Component: generator --------------------------------------------
    MetricSpec("component_generator.avg_faithfulness", "higher_better", "critical",
               "Generator groundedness when fed the correct context directly"),
    MetricSpec("component_generator.avg_answer_relevance", "higher_better", "important",
               "Generator on-topic-ness when fed the correct context directly"),
    MetricSpec("component_generator.avg_correctness", "higher_better", "important",
               "Generator correctness vs. reference answers"),

    # --- Pipeline: RAG Triad ----------------------------------------------
    MetricSpec("pipeline_rag_triad.avg_context_relevance", "higher_better", "important",
               "Whether retrieved context actually relates to the question"),
    MetricSpec("pipeline_rag_triad.avg_faithfulness", "higher_better", "critical",
               "End-to-end groundedness of the assembled pipeline"),
    MetricSpec("pipeline_rag_triad.avg_answer_relevance", "higher_better", "critical",
               "End-to-end on-topic-ness of the assembled pipeline"),

    # --- Safety: scope adherence -------------------------------------------
    MetricSpec("scope_adherence.pass_rate", "higher_better", "critical",
               "Whether the assistant stays within its intended scope"),

    # --- Application: quality ------------------------------------------
    MetricSpec("application_level.quality.avg_correctness", "higher_better", "important",
               "End-to-end correctness vs. reference answers"),

    # --- Application: safety -----------------------------------------------
    MetricSpec("application_level.safety.toxicity_violations", "lower_better", "critical",
               "Count of toxic outputs across the golden dataset"),
    MetricSpec("application_level.safety.pii_violations", "lower_better", "critical",
               "Count of PII-leaking outputs across the golden dataset"),
    MetricSpec("application_level.safety.jailbreak_failures", "lower_better", "critical",
               "Count of jailbreak probes the system failed to resist"),

    # --- Application: ops --------------------------------------------------
    MetricSpec("application_level.ops.success_rate", "higher_better", "critical",
               "Fraction of pipeline calls that completed without error"),
    MetricSpec("application_level.ops.latency.total_p95_seconds", "lower_better", "minor",
               "95th-percentile end-to-end response latency"),
    MetricSpec("application_level.ops.latency.total_p99_seconds", "lower_better", "minor",
               "99th-percentile end-to-end response latency"),
    MetricSpec("application_level.ops.cost.avg_cost_usd_per_query", "lower_better", "minor",
               "Average estimated cost per query"),
]

METRIC_REGISTRY_BY_PATH: dict[str, MetricSpec] = {m.path: m for m in METRIC_REGISTRY}
