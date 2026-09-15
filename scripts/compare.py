#!/usr/bin/env python3
"""
Compares reports/candidate.json against reports/baseline.json metric by
metric, and classifies every metric as IMPROVED / REGRESSED / NOISE
(unchanged within the noise-threshold band) / UNCHANGED.

    python scripts/compare.py \
        --baseline reports/baseline.json \
        --candidate reports/candidate.json \
        --noise reports/noise_thresholds.json \
        --out reports/comparison.json

If --noise is omitted or the file doesn't exist, every non-zero delta is
treated as significant (zero tolerance) -- a stricter but always-safe
fallback. Direction (higher/lower is better) comes from
evals/metric_registry.py, so e.g. a latency INCREASE and a faithfulness
DECREASE are both correctly classified as regressions even though the sign
of the raw delta is opposite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.metric_registry import METRIC_REGISTRY_BY_PATH  # noqa: E402


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_delta(metric: str, baseline_value: float, candidate_value: float, threshold: float) -> tuple[str, float]:
    spec = METRIC_REGISTRY_BY_PATH.get(metric)
    direction = spec.direction if spec else "higher_better"
    raw_delta = candidate_value - baseline_value

    # Normalize so "favorable_delta > 0" always means "got better", regardless
    # of whether the raw metric wants to go up (faithfulness) or down (latency).
    favorable_delta = raw_delta if direction == "higher_better" else -raw_delta

    if abs(favorable_delta) <= threshold:
        status = "NOISE" if threshold > 0 else "UNCHANGED"
    elif favorable_delta > 0:
        status = "IMPROVED"
    else:
        status = "REGRESSED"
    return status, raw_delta


def compare(baseline_path: str, candidate_path: str, noise_path: str | None) -> dict:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    noise = _load(noise_path)["noise"] if noise_path and os.path.exists(noise_path) else {}

    rows = []
    all_metrics = sorted(set(baseline["metrics"]) | set(candidate["metrics"]))
    for metric in all_metrics:
        if metric not in baseline["metrics"]:
            rows.append({"metric": metric, "status": "NEW_IN_CANDIDATE", "baseline": None,
                         "candidate": candidate["metrics"][metric], "delta": None, "tier": _tier(metric)})
            continue
        if metric not in candidate["metrics"]:
            rows.append({"metric": metric, "status": "MISSING_IN_CANDIDATE", "baseline": baseline["metrics"][metric],
                         "candidate": None, "delta": None, "tier": _tier(metric)})
            continue

        b_val = baseline["metrics"][metric]
        c_val = candidate["metrics"][metric]
        threshold = noise.get(metric, {}).get("threshold", 0.0)
        status, delta = classify_delta(metric, b_val, c_val, threshold)
        rows.append({
            "metric": metric, "status": status, "baseline": b_val, "candidate": c_val,
            "delta": delta, "noise_threshold": threshold, "tier": _tier(metric),
        })

    regressions = [r for r in rows if r["status"] == "REGRESSED"]
    improvements = [r for r in rows if r["status"] == "IMPROVED"]

    return {
        "baseline_captured_at_unix": baseline.get("captured_at_unix"),
        "candidate_captured_at_unix": candidate.get("captured_at_unix"),
        "baseline_config": baseline.get("config"),
        "candidate_config": candidate.get("config"),
        "rows": rows,
        "n_regressed": len(regressions),
        "n_improved": len(improvements),
        "n_critical_regressed": sum(1 for r in regressions if r["tier"] == "critical"),
    }


def _tier(metric: str) -> str:
    spec = METRIC_REGISTRY_BY_PATH.get(metric)
    return spec.tier if spec else "unregistered"


def print_report(result: dict) -> None:
    print(f"{'METRIC':<55} {'TIER':<10} {'STATUS':<12} {'BASELINE':>12} {'CANDIDATE':>12} {'DELTA':>12}")
    print("-" * 115)
    for r in result["rows"]:
        b = f"{r['baseline']:.4f}" if r["baseline"] is not None else "-"
        c = f"{r['candidate']:.4f}" if r["candidate"] is not None else "-"
        d = f"{r['delta']:+.4f}" if r["delta"] is not None else "-"
        marker = {"REGRESSED": "❌", "IMPROVED": "✅", "NOISE": "▫️", "UNCHANGED": "▫️"}.get(r["status"], "?")
        print(f"{r['metric']:<55} {r['tier']:<10} {marker} {r['status']:<10} {b:>12} {c:>12} {d:>12}")
    print("-" * 115)
    print(f"Regressed: {result['n_regressed']} (critical: {result['n_critical_regressed']}) | "
          f"Improved: {result['n_improved']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="reports/baseline.json")
    parser.add_argument("--candidate", default="reports/candidate.json")
    parser.add_argument("--noise", default="reports/noise_thresholds.json")
    parser.add_argument("--out", default="reports/comparison.json")
    args = parser.parse_args()

    result = compare(args.baseline, args.candidate, args.noise)
    print_report(result)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
