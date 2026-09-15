#!/usr/bin/env bash
# Reproduces the regression-testing demo in reports/demo_regression_workflow/:
# capture a baseline, measure noise, deliberately degrade TOP_K, capture a
# candidate, compare, and get a promotion verdict. Exits non-zero if the
# (expected) BLOCK verdict fires, matching what CI would do.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Capture baseline (TOP_K=3, the default) ==="
python3 scripts/capture_metrics.py --out reports/baseline.json

echo ""
echo "=== 2. Measure noise (5 runs, unchanged config) ==="
python3 scripts/measure_noise.py --runs 5
rm -f /tmp/_noise_run_*.json

echo ""
echo "=== 3. Simulate a code change: TOP_K 3 -> 1 ==="
TOP_K=1 python3 scripts/capture_metrics.py --out reports/candidate.json

echo ""
echo "=== 4. Compare candidate vs baseline ==="
python3 scripts/compare.py

echo ""
echo "=== 5. Promotion decision ==="
set +e
python3 scripts/promote.py
verdict_exit=$?
set -e

echo ""
echo "(promote.py exited $verdict_exit -- 1 means BLOCK, matching CI's expected behavior for this demo)"
exit "$verdict_exit"
