#!/usr/bin/env python3
"""
Estimates how much each metric naturally wobbles between runs with NOTHING
changed, so scripts/compare.py can tell a real regression apart from
ordinary LLM-as-judge / API noise.

Method (per the RAG Regression Testing noise-threshold approach):
  1. Run the full suite NOISE_RUNS times (default 5) with no code changes.
  2. Compute the standard deviation of each metric across those runs.
  3. Save `noise_sigma_multiplier * std` (default 2 sigma) per metric as the
     tolerance band -- any candidate-vs-baseline delta smaller than this is
     treated as noise, not a regression.

    python scripts/measure_noise.py --runs 5 --out reports/noise_thresholds.json

Honesty note: in the shipped mock mode, every provider is fully
deterministic (same input -> same output every time), so measured noise
here will legitimately be ~0 for every metric. That's not a bug -- it's the
correct answer for a deterministic system. The mechanism becomes load-bearing
the moment a real LLM/judge is in the loop, where run-to-run variance is
real and this script is what stops that variance from being misread as a
regression. See README "Regression testing workflow".
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.capture_metrics import capture  # noqa: E402
from src.config import SETTINGS  # noqa: E402


def measure(n_runs: int) -> dict:
    per_metric_values: dict[str, list[float]] = {}
    for i in range(n_runs):
        print(f"  noise run {i + 1}/{n_runs} ...")
        snapshot = capture(f"/tmp/_noise_run_{i}.json")
        for metric, value in snapshot["metrics"].items():
            per_metric_values.setdefault(metric, []).append(value)

    noise = {}
    for metric, values in per_metric_values.items():
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        noise[metric] = {
            "mean": mean,
            "std": std,
            "threshold": SETTINGS.noise_sigma_multiplier * std,
            "n_runs": len(values),
            "raw_values": values,
        }
    return noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=SETTINGS.noise_runs)
    parser.add_argument("--out", default="reports/noise_thresholds.json")
    args = parser.parse_args()

    print(f"Measuring noise over {args.runs} runs with LLM_PROVIDER={SETTINGS.llm_provider!r} "
          f"(no code/config changes between runs) ...")
    noise = measure(args.runs)

    payload = {"measured_at_unix": time.time(), "n_runs": args.runs, "sigma_multiplier": SETTINGS.noise_sigma_multiplier,
               "noise": noise}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote noise thresholds for {len(noise)} metrics to {args.out}:")
    for metric, stats in sorted(noise.items()):
        print(f"  {metric}: mean={stats['mean']:.4f} std={stats['std']:.4f} threshold=±{stats['threshold']:.4f}")


if __name__ == "__main__":
    main()
