"""Turns the nested dicts each eval suite returns into one flat
{metric_path: value} dict, using METRIC_REGISTRY as the list of paths to
pull. This flat shape is what gets written to baseline.json/candidate.json
and diffed by scripts/compare.py."""
from __future__ import annotations

from evals.metric_registry import METRIC_REGISTRY


def _get_path(d: dict, dotted_path: str):
    node = d
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def flatten_metrics(suite_results: dict[str, dict]) -> dict[str, float]:
    """`suite_results` maps suite name (e.g. "component_retriever") to that
    suite's raw result dict. Returns {registry_path: value} for every
    registered metric found; metrics whose suite wasn't run are skipped
    rather than defaulted to 0, so a missing suite doesn't silently look
    like a metric of 0."""
    flat: dict[str, float] = {}
    for spec in METRIC_REGISTRY:
        suite_name, rest = spec.path.split(".", 1)
        suite_result = suite_results.get(suite_name)
        if suite_result is None:
            continue
        value = _get_path(suite_result, rest)
        if value is not None:
            flat[spec.path] = float(value)
    return flat
