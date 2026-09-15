# Regression pipeline demo — a real, reproduced regression catch

These files are **not synthetic** — they're the actual output of running the
regression-testing workflow once, on purpose, to prove the mechanism works
rather than just asserting that it does. Reproduce with:

```bash
./scripts/demo_regression_workflow.sh
```

## What was done

1. `baseline.json` — captured the full metric suite with the default config
   (`TOP_K=3`).
2. `noise_thresholds.json` — ran the suite 5 more times with **nothing**
   changed, to measure how much each metric naturally wobbles run-to-run.
   In mock mode every provider is deterministic, so every threshold here is
   legitimately ≈0 — that's the correct answer for a deterministic system,
   not a bug (see the script's docstring for what changes with a real LLM).
3. `candidate.json` — captured the suite again after one deliberate change:
   `TOP_K=1` instead of 3 (retrieve only the single best chunk instead of
   the top 3).
4. `comparison.json` — `scripts/compare.py` diffed candidate vs. baseline.
5. `promotion_verdict.json` — `scripts/promote.py` turned that diff into a
   deploy decision.

## What it found

Dropping `TOP_K` from 3 to 1 is exactly the kind of change that *looks* like
a win in isolation:

| Metric | Baseline | Candidate | Change |
|---|---|---|---|
| Retriever precision@K | 0.36 | 0.63 | ✅ improved |
| P95 / P99 latency | ~0.10ms | ~0.09ms | ✅ improved |
| **Retriever recall@K** | 0.81 | 0.63 | ❌ **regressed** |
| **RAG Triad answer relevance** | 0.97 | 0.93 | ❌ **regressed** |
| **Scope adherence pass rate** | 0.90 | 0.80 | ❌ **regressed** |

If you only looked at precision and latency, you'd ship this change and call
it an optimization. The regression suite caught that recall — a
**critical**-tier metric in `evals/metric_registry.py` — dropped by more
than the noise threshold, and that the drop cascaded into the pipeline-level
RAG Triad and scope-adherence suites. `promote.py`'s verdict:

```
🛑 VERDICT: BLOCK
Reason: 3 critical metric(s) regressed beyond noise threshold:
component_retriever.avg_recall_at_k, pipeline_rag_triad.avg_answer_relevance,
scope_adherence.pass_rate
```

Exit code `1` — this is exactly what would fail a CI check and block a PR.
This is the concrete answer to "how do you know your regression gate
actually works": it caught a real regression, on a real (if deliberately
induced) code change, and it did so specifically because the metric was
tiered `critical` in the registry rather than every metric being treated
identically.
