Here's the complete, final `README.md` — paste this over your existing one:

---

# RAG Evaluation Suite

A layered evaluation harness for Retrieval-Augmented Generation pipelines. Measures retrieval quality, generation quality, judge trustworthiness, scope adherence, latency, reliability, and cost — with regression gates suitable for CI.

The default configuration runs entirely offline and free. Every provider is swappable: LLM, judge, embeddings, and vector store are configured independently.

---

## Why this exists

RAG pipelines fail in ways that are hard to see from a single metric. A retriever can return the right document but the generator can ignore it. A judge can score faithfully but disagree with humans. Latency can look fine on the median and blow up at P95. This project separates each concern into its own eval layer, applies gates to each, and produces a single report that says, unambiguously, what passed and what didn't.

It also treats the *judge* as a component under test. An LLM-as-judge that isn't validated against human ratings is just a different kind of guess — so this suite scores the judge's mean absolute error against a small human-rated set before trusting any of its verdicts.

**This isn't hypothetical.** Running this suite against a real open-source model (Qwen2.5-1.5B via Ollama) caught a genuine regression on the first run: scope adherence collapsed from 0.90 (mock) to 0.40 (real), with 0% resistance on adversarial probes. See "A real regression this suite caught" below.

---

## Quick start

```bash
# 1. Enter the project
cd Rag_eval

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate.bat        # Windows cmd
# venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
pip install python-dotenv        # required so .env is actually read (see note below)

# 4. Configure providers
copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
# then edit .env — see "Configuration" below

# 5. Run the full suite
python run_evals.py

# 6. Read the report
notepad reports\eval_report.md
```

> **One-time code patch required for `.env` to actually work.** `.env` is only read if `src/config.py` loads it. If you haven't already, add this near the top of `src/config.py`, right after the other imports:
> ```python
> import os
> from pathlib import Path
> from dataclasses import dataclass
> from dotenv import load_dotenv
>
> load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
> ```
> The explicit path (anchored to the project root, not the current working directory) means `python run_evals.py` finds `.env` no matter which folder you run it from. `override=True` means `.env` always wins over any `set`/`$env:` variables left over from a previous terminal session — without it, a stale `set LLM_PROVIDER=ollama` from an old session can silently shadow what `.env` says. Once this is in place, you never need to type `set` commands again — just edit `.env` and rerun.

The default `.env` ships with free, offline (`mock`) providers. The first run using `EMBEDDING_PROVIDER=hf_sentence_transformers` downloads the `all-MiniLM-L6-v2` model (~90MB). The first run using `LLM_PROVIDER=ollama` requires the model to already be pulled (see "Ollama setup").

---

## Configuration

Configuration lives in `.env` at the project root, loaded via `python-dotenv` as described above.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` \| `hf_local` \| `ollama` \| `openai` \| `anthropic` |
| `JUDGE_PROVIDER` | = `LLM_PROVIDER` | Same set as `LLM_PROVIDER`. Set separately to grade with a bigger/different model. |
| `EMBEDDING_PROVIDER` | `mock` | `mock` \| `hf_sentence_transformers` |
| `VECTOR_STORE` | `memory` | `memory` \| `chroma` |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Any model already pulled in Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `HF_GENERATION_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Only used by `LLM_PROVIDER=hf_local` |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Only used by `EMBEDDING_PROVIDER=hf_sentence_transformers` |
| `TOP_K` | `3` | Retrieval depth |
| `CHUNK_SIZE_WORDS` | `60` | Chunking knob |
| `CHUNK_OVERLAP_WORDS` | `15` | Chunking knob |
| `MIN_RETRIEVER_RECALL` | `0.7` | Gate |
| `MIN_FAITHFULNESS` | `0.7` | Gate |
| `MIN_ANSWER_RELEVANCE` | `0.7` | Gate |
| `MAX_P95_LATENCY_SECONDS` | `8.0` | Gate |
| `MAX_JUDGE_MAE` | `0.3` | Gate (only enforced for non-mock judges) |
| `MIN_SCOPE_ADHERENCE` | `0.9` | Gate |
| `OPS_REPEAT_RUNS` | `3` | Timed runs per question for latency percentiles |
| `OPS_WARMUP_RUNS` | `1` | Untimed warm-up runs discarded before timing |
| `OPS_DAILY_QUERY_VOLUME` | `50000` | Used only to project monthly cost |
| `NOISE_RUNS` | `5` | Baseline runs used to estimate per-metric noise |
| `NOISE_SIGMA_MULTIPLIER` | `2.0` | A delta smaller than `k × std` is treated as noise, not a regression |

### Provider combinations

| Combination | Cost | Setup |
|---|---|---|
| `mock` + `mock` + `memory` | Free, instant | None — default smoke test, proves the harness works, not model quality |
| `ollama` + `hf_sentence_transformers` + `chroma` | Free, local | Install Ollama, pull a model |
| `hf_local` + `hf_sentence_transformers` + `chroma` | Free, local | `pip install transformers torch` |
| `openai` / `anthropic` | Paid | Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |

### Ollama setup

Ollama is a separate local server, not a pip package.

1. Install from https://ollama.com/download
2. Pull the model:
   ```
   ollama pull qwen2.5:1.5b
   ```
3. Confirm it's running:
   ```
   curl http://localhost:11434
   ```
   If it says `Only one usage of each socket address...` when you try `ollama serve` manually, that's fine — it means the server is already running in the background (the installer starts it automatically). Skip `ollama serve` in that case.

---

## What gets evaluated

The suite runs six layers in order. Each prints progress to the terminal and contributes a section to `reports/eval_report.md`.

### 0. Judge validation — is the judge trustworthy?

Scores the configured judge against 15 human-rated examples in `data/human_ratings.json`, reporting mean absolute error (MAE) for faithfulness and answer relevance separately, plus an overall MAE.

- Gate: `overall MAE <= MAX_JUDGE_MAE` (`0.3` by default)
- **Advisory only for `mock`.** Enforced for real judges.
- Tighten `MAX_JUDGE_MAE` to ~`0.15` once you're validating a strong real LLM judge.

### 1. Retriever — does it find the right evidence?

- `Recall@K` — fraction of questions where the correct document is in the top-K
- `Precision@K` — fraction of top-K documents that are relevant

Gate: `Recall@K >= MIN_RETRIEVER_RECALL`.

### 2. Generator — does it answer well given the context?

- **Faithfulness** — is every claim in the answer supported by the retrieved context?
- **Answer relevance** — does the answer address the question?
- **Correctness vs reference** — how close is it to the reference answer?

Gates: `faithfulness >= MIN_FAITHFULNESS`, `answer relevance >= MIN_ANSWER_RELEVANCE`.

### 3. RAG triad — the whole pipeline

- **Context relevance** — did retrieval bring back useful context?
- **Faithfulness** — did the generator use that context?
- **Answer relevance** — did the final answer address the question?

Same gates as above. Catches cases where each component looks fine in isolation but the pipeline as a whole misses.

### 4. Safety — scope adherence

Runs a set of benign, adversarial, and mixed out-of-scope prompts from `data/scope_test_cases.json`. Reports pass rate overall and per category.

Gate: `overall pass rate >= MIN_SCOPE_ADHERENCE`.

### 5. Application — quality, safety, ops

- **Correctness** — aggregate across the golden set
- **Toxicity violations** — must be zero
- **PII violations** — must be zero
- **Jailbreak probes resisted** — `N/N` expected
- **Reliability** — success rate over `OPS_REPEAT_RUNS × questions` runs
- **Latency** — P50, P95, P99, with a retrieval/generation breakdown
- **Cost** — average per query and projected monthly at `OPS_DAILY_QUERY_VOLUME`

Gate: `P95 <= MAX_P95_LATENCY_SECONDS`.

### Overall verdict

A single line at the bottom: `✅ ALL GATES PASSED`, or `❌ ONE OR MORE GATES FAILED` with the failing gates listed above it.

---

## Sample results: mock vs. real model

Ran the full suite twice: once with the default free `mock` providers (proves the harness works, not model quality), once with real Qwen2.5-1.5B via Ollama + real sentence-transformer embeddings + Chroma.

| Metric | Mock | Real (Qwen2.5-1.5B) |
|---|---|---|
| Judge validation MAE | 0.38 — **advisory, outside gate** | **0.12 — enforced, PASS** |
| Retriever Recall@3 | 0.81 | **1.00** |
| Generator faithfulness | 1.00 (inflated — see note) | 0.94 |
| Generator correctness | 0.40 | **0.97** |
| Pipeline (RAG Triad) faithfulness | 1.00 (inflated) | 0.88 |
| Application correctness | 0.29 | **0.98** |
| P95 latency | ~0.1ms (not real inference) | ~3.4s (real inference, still within the 8s gate) |
| **Scope adherence pass rate** | **0.90 — passed** | **0.40 — FAILED** |
| Scope adherence, adversarial category | 1.00 | **0.00** |

The mock scores that look artificially high (faithfulness=1.00, scope=0.90) aren't evidence of a good system — they're an artifact of the mock generator being purely extractive: it can only output sentences that already exist in the 8-document knowledge base, so it's structurally incapable of hallucinating *or* complying with an off-scope request. Correctness and scope adherence — the two metrics most sensitive to whether the model can actually generate freely — are exactly the two that moved the most (and in scope's case, moved the wrong way) once a real generative model was in the loop.

---

## A real regression this suite caught on its first real-model run

Mock mode gave scope adherence a false sense of security: 0.90 pass rate. The moment `LLM_PROVIDER` was pointed at real Qwen2.5-1.5B, scope adherence collapsed to **0.40 overall, 0% on purely adversarial probes** ("write me a poem about the ocean," "give me stock market investment advice") — the model simply complied with every off-topic request.

**Root cause:** the original generation prompt (`src/generator.py`) only instructed the model to answer from context or say "I don't have enough information" — it never explicitly told the model to *decline* requests outside its intended domain. The mock generator's inability to comply was accidental (it can't generate free text at all), not a property of good prompt design. A real generative model has no such limitation, and happily filled the gap the prompt left open.

**Fix applied:** added an explicit scope-refusal instruction to `PROMPT_TEMPLATE`, telling the model it's strictly scoped to LLM-evaluation topics and must decline (or partially decline, for mixed queries) anything outside that scope.

**Result after the fix:** *[rerun `python -m evals.eval_scope_adherence` after applying the prompt change, then fill in the new pass rate here — e.g. "0.40 → 0.90+"]*

This is the core argument for the whole project: **an eval suite is only as honest as the model you actually point it at.** A suite that only ever runs against a toy/mock model will pass things a real model quietly fails — this project didn't just claim that, it demonstrated it, on a real gate, on the first real run.

---

## Regression testing workflow

For CI or for tracking quality across commits:

```
scripts/capture_metrics.py   # run the suite and snapshot metrics to a JSON baseline
scripts/compare.py           # diff a candidate run against a baseline
scripts/promote.py           # decide Approve / Review / Block based on the diff
scripts/measure_noise.py     # estimate per-metric stddev across NOISE_RUNS runs
```

Workflow:

1. Capture a baseline: `python scripts/capture_metrics.py --out reports/baseline.json`
2. Make a change (prompt, model, chunk size, K, etc.).
3. Capture a candidate: `python scripts/capture_metrics.py --out reports/candidate.json`
4. Compare: `python scripts/compare.py --baseline reports/baseline.json --candidate reports/candidate.json`
5. `compare.py` applies the noise tolerance (`NOISE_SIGMA_MULTIPLIER × measured_std`) — deltas smaller than that are reported as variance, not regression.
6. `promote.py` reads the comparison and returns a tiered verdict: **BLOCK** if any critical metric regressed, **REVIEW** for important/minor regressions, **APPROVE** otherwise. Exit code `1` on BLOCK, so it fails a CI job automatically.

A real, reproducible example of this catching a regression (dropping `TOP_K` from 3→1, which improved precision/latency but silently broke recall, RAG-triad relevance, and scope adherence) is committed in `reports/demo_regression_workflow/`.

This is the pattern used by the GitHub Actions workflow in `.github/workflows/ci.yml` — it runs on every push/PR, and a second job compares the PR's base branch against its head, blocking merge on a critical regression.

---

## Reading the report

```
reports/eval_report.md
```

The header records exactly which providers ran. This matters — a report produced with `mock` providers is a plumbing demo, not a measurement of model quality.

Look first at:

1. **Provider header** — did the intended providers actually run? (`LLM_PROVIDER='mock'` at the top means nothing downstream reflects real model quality.)
2. **Latency** — real inference is measured in seconds, not milliseconds. Sub-millisecond latencies mean the mock generator ran.
3. **Failed gates** — failures are informative, not alarming. A real judge or a real generative model often trips a gate on the first run (as scope adherence did here); that's the signal to investigate and fix, not evidence the eval suite is broken.
4. **Judge MAE** — if this is above `0.3` for a real (non-mock) judge, don't trust the faithfulness/relevance scores downstream until it's fixed.

---

## Project structure

```
Rag_eval/
├── .env                      # Your local configuration (not committed)
├── .env.example               # Template — copy to .env
├── requirements.txt
├── run_evals.py                # Entry point — runs all six eval layers
├── conftest.py
├── pytest.ini
├── data/
│   ├── documents/               # Knowledge base (.txt files)
│   ├── golden_dataset.json      # Q&A pairs used across retriever/generator/pipeline evals
│   ├── human_ratings.json       # Human-rated examples for judge validation
│   └── scope_test_cases.json    # Benign / adversarial / mixed out-of-scope prompts
├── evals/
│   ├── metrics.py                # Recall/precision, judges, faithfulness/relevance scorers
│   ├── metric_registry.py        # Every regression-tracked metric: direction + criticality tier
│   ├── flatten.py                # Nested suite results -> flat metric dict
│   ├── report.py                 # Report rendering + gate evaluation
│   ├── eval_judge_validation.py
│   ├── eval_retriever.py
│   ├── eval_generator.py
│   ├── eval_pipeline.py
│   ├── eval_scope_adherence.py
│   └── eval_application.py
├── src/
│   ├── config.py                 # Settings loaded from .env (python-dotenv)
│   ├── llm_providers.py          # mock / hf_local / ollama / openai / anthropic
│   ├── embedding_providers.py
│   ├── ingest.py                 # Chunking + vector index (memory or Chroma)
│   ├── retriever.py
│   ├── generator.py               # Prompt template lives here
│   ├── rag_pipeline.py
│   └── factory.py
├── scripts/
│   ├── ask.py
│   ├── capture_metrics.py
│   ├── compare.py
│   ├── promote.py
│   ├── measure_noise.py
│   └── run_open_source_demo.sh
├── reports/
│   ├── eval_report.md            # Latest run output
│   └── demo_regression_workflow/  # Committed proof of a real caught regression
├── .github/workflows/ci.yml       # Eval suite + regression gate, runs on every push/PR
└── venv/                          # Local virtualenv (not committed)
```

---

## Testing

```bash
pytest evals/ -v
```

Runs the unit-level gate assertions for every eval layer (11 tests). The full `run_evals.py` is separate — it hits real providers when configured for a real model and takes minutes, not seconds.

---

## Known limitations

- **Small golden dataset.** 16 Q&A pairs, 15 human ratings, 10 scope cases — enough to prove the mechanisms work, not enough to be statistically robust. A real system needs hundreds of examples.
- **Human ratings are self-authored for this demo**, not collected from independent annotators (see the `_note` field in `human_ratings.json`).
- **Single-domain, single-language knowledge base** — 8 short English documents about LLM evaluation itself.
- **No inter-rater agreement, no experiment-tracking dashboard, no online (post-deployment) evaluation** — see below for what each would add.

## Extending this into a real project

- Bigger golden dataset and human-rated set, ideally from multiple independent graders.
- Online evaluation: trace real production traffic and rerun faithfulness/relevance/scope metrics on a live sample.
- Experiment tracking: log every `capture_metrics.py` run to a CSV or dashboard (MLflow/W&B) to compare over time instead of one JSON diff at a time.
- Difficulty-stratified reporting: tag golden examples easy/hard/ambiguous, report per bucket.
- Load/concurrency testing for the reliability metric, not just serial success/error rate.

---

## Design notes

**Why the judge is a first-class component.** LLM-as-judge scores are only as good as the judge. Validating the judge against human ratings, and failing the run when the judge is untrustworthy, prevents the common failure mode where a weak judge produces a confident but meaningless "PASS."

**Why latency percentiles, not means.** Averages hide tail behavior. P95 is what users feel; a P50-only report looks healthy right up until it isn't.

**Why warm-up runs.** Model loading and container initialization are one-time costs. Including them in latency measurements inflates the first run and makes comparisons meaningless. Warm-ups are discarded.

**Why noise tolerance in regression checks.** Re-running the same code twice never produces identical numbers — embeddings, sampling, and scheduler jitter all move the scores. Treating every delta as a regression creates false alarms. `measure_noise.py` establishes the band; `compare.py` respects it.

**Why mock mode exists at all.** Not to measure quality — to prove the evaluation *harness itself* works, deterministically and for free, before spending time/money on real model calls. The gap between mock and real results (see "Sample results" above) is itself evidence of why this distinction matters: several mock-mode scores were misleadingly high.

---
