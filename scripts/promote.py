#!/usr/bin/env python3
"""
Automated deployment verdict, based on scripts/compare.py's output and each
metric's criticality tier (evals/metric_registry.py):

  BLOCK   - any CRITICAL metric regressed beyond the noise threshold.
            Deployment is halted; this is the same signal
            .github/workflows/ci.yml uses to fail a PR check.
  REVIEW  - an IMPORTANT (non-critical) metric regressed. Not blocking, but
            flagged for a human to look at before merging.
  APPROVE - nothing regressed beyond noise, or only MINOR metrics moved.

    python scripts/promote.py --comparison reports/comparison.json

Exit code: 0 for APPROVE or REVIEW, 1 for BLOCK -- matching how you'd wire
this into CI (BLOCK fails the job, REVIEW passes but a human still sees the
flagged output in the job log / PR comment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def decide(comparison: dict) -> dict:
    regressed = [r for r in comparison["rows"] if r["status"] == "REGRESSED"]
    critical = [r for r in regressed if r["tier"] == "critical"]
    important = [r for r in regressed if r["tier"] == "important"]
    minor = [r for r in regressed if r["tier"] == "minor"]

    if critical:
        verdict = "BLOCK"
        reason = f"{len(critical)} critical metric(s) regressed beyond noise threshold: " + ", ".join(
            r["metric"] for r in critical
        )
    elif important:
        verdict = "REVIEW"
        reason = f"{len(important)} important metric(s) regressed beyond noise threshold: " + ", ".join(
            r["metric"] for r in important
        )
    elif minor:
        verdict = "REVIEW"
        reason = f"{len(minor)} minor metric(s) regressed beyond noise threshold: " + ", ".join(
            r["metric"] for r in minor
        )
    else:
        verdict = "APPROVE"
        reason = "No metric regressed beyond its noise threshold."

    return {
        "verdict": verdict,
        "reason": reason,
        "n_critical_regressed": len(critical),
        "n_important_regressed": len(important),
        "n_minor_regressed": len(minor),
        "n_improved": comparison["n_improved"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="reports/comparison.json")
    parser.add_argument("--out", default="reports/promotion_verdict.json")
    args = parser.parse_args()

    with open(args.comparison, "r", encoding="utf-8") as f:
        comparison = json.load(f)

    verdict = decide(comparison)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    icon = {"APPROVE": "✅", "REVIEW": "⚠️", "BLOCK": "🛑"}[verdict["verdict"]]
    print(f"{icon} VERDICT: {verdict['verdict']}")
    print(f"Reason: {verdict['reason']}")
    print(f"Wrote {args.out}")

    return 1 if verdict["verdict"] == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
