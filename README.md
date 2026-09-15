# RAG Evaluation Suite

A layered evaluation harness for Retrieval-Augmented Generation pipelines.
Measures retrieval quality, generation quality, judge trustworthiness,
scope adherence, latency, reliability, and cost — with regression gates
suitable for CI.

The default configuration runs entirely offline and free. Every provider
is swappable: LLM, judge, embeddings, and vector store are configured
independently, including using a *different* model for judging than for
generation.

---

## Why this exists

RAG pipelines fail in ways that are hard to see from a single metric. A
retriever can return the right document but the generator can ignore it.
A judge can score faithfully but disagree with humans. Latency can look
fine on the median and blow up at P95. This project separates each
concern into its own eval layer, applies gates to each, and produces a
single report that says, unambiguously, what passed and what didn't.

It also treats the *judge itself* as a component under test — not once,
but repeatedly. An LLM-as-judge that isn't validated against ground
truth is just a different kind of guess. This project doesn't just claim
that; it demonstrates it, across three separate real debugging cycles
documented below, where a judge was found to be unreliable, diagnosed,
and fixed.

---

## Quick start

\`\`\`bash
cd Rag_eval
python -m venv venv
venv\Scripts\activate.bat        # Windows cmd
pip install -r requirements.txt
pip install python-dotenv

copy .env.example .env
# edit .env — see "Configuration" below

python run_evals.py
notepad reports\eval_report.md
\`\`\`

> **Note:** `.env` is loaded via `python-dotenv` in `src/config.py`
> (`load_dotenv(path, override=True)`, anchored to the project root).
> Edit `.env` and rerun — no need to `set` environment variables manually.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` \| `hf_local` \| `ollama` \| `openai` \| `anthropic` |
| `JUDGE_PROVIDER` | = `LLM_PROVIDER` | Model used for LLM-as-judge scoring |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Generation model |
| `OLLAMA_JUDGE_MODEL` | = `OLLAMA_MODEL` | **Separate, optionally larger model used only for judging** — e.g. run generation on `qwen2.5:1.5b` (fast, cheap) and judging on `qwen2.5:7b` (slower, more reliable) |
| `EMBEDDING_PROVIDER` | `mock` | `mock` \| `hf_sentence_transformers` |
| `VECTOR_STORE` | `memory` | `memory` \| `chroma` |
| `TOP_K` | `3` | Retrieval depth |
| `MIN_RETRIEVER_RECALL` | `0.7` | Gate |
| `MIN_FAITHFULNESS` | `0.7` | Gate |
| `MIN_ANSWER_RELEVANCE` | `0.7` | Gate |
| `MAX_P95_LATENCY_SECONDS` | `8.0` | Gate |
| `MAX_JUDGE_MAE` | `0.3` | Gate (enforced only for non-mock judges) |
| `MIN_SCOPE_ADHERENCE` | `0.9` | Gate |

Full variable list in `.env.example`.

---

## What gets evaluated (six layers)

0. **Judge validation** — MAE against `data/human_ratings.json`. Advisory for mock, enforced for real judges.
1. **Retriever** — Recall@K, Precision@K
2. **Generator** — Faithfulness, Answer Relevance, Correctness (fed known-correct context directly, isolated from retrieval)
3. **RAG Triad** — Context Relevance, Faithfulness, Answer Relevance on the assembled pipeline
4. **Scope adherence** — benign / adversarial / mixed queries, judged by an LLM (`evals/eval_scope_adherence.py`), not keyword matching
5. **Application** — correctness, toxicity, PII, jailbreak resistance, latency percentiles, reliability, cost

---

## Sample results: mock vs. real model

| Metric | Mock | Real (Qwen2.5-1.5B, sentence-transformers, Chroma) |
|---|---|---|
| Judge validation MAE | 0.38 — advisory, outside gate | **0.12 — enforced, PASS** |
| Retriever Recall@3 | 0.81 | **1.00** |
| Generator correctness | 0.40 | **0.97** |
| Application correctness | 0.29 | **0.98** |
| P95 latency | ~0.1ms (not real inference) | ~3.4s (real inference, within the 8s gate) |

---

## The scope-adherence debugging story

This is the most useful part of the project to actually walk through in
an interview, because it's not a clean pass — it's a real, multi-round
debugging log.

**Round 1 — keyword-based scope check, real Qwen2.5-1.5B generator.**
Pass rate: 0.40. Investigation showed 4 of 6 "failures" were false
positives: the model correctly refused off-topic requests but was
penalized for naming the declined topic while refusing (e.g. *"I can't
write about the ocean"* flagged for containing "ocean"). Two were real
failures: the model wrote working bubble-sort code, and fully wrote a
requested anniversary message after correctly answering an unrelated
in-scope question first.

**Round 2 — refined keyword check (refusal-marker + length heuristic).**
Pass rate: 0.80. Correctly stopped penalizing clean refusals; the two
genuine violations (code generation, mixed-query full compliance)
remained correctly flagged. Root cause of the original two failures:
`PROMPT_TEMPLATE` in `src/generator.py` never explicitly told the model
to decline out-of-scope requests, even partially, so a real generative
model filled that gap.

**Round 3 — replaced keyword matching with an LLM judge
(`qwen2.5:1.5b`).** Pass rate collapsed to 0.40, but for a new reason:
the judge began confidently *hallucinating* violations that didn't
exist in the text — e.g. scoring a clean refusal as a violation and
inventing the claim that a poem had been written, when it hadn't. This
is a distinct and more concerning failure mode than the keyword check's
false positives: a small model confidently asserting something false
about the very text in front of it.

**Round 4 — swapped judge model to `qwen2.5:7b`
(`OLLAMA_JUDGE_MODEL`), same generator.** Pass rate: 0.50. The
hallucination problem was gone — reasoning text now accurately described
what was actually in each answer. But a new, narrower bug appeared: the
7B judge conflated "the assistant declined to help" with "the assistant
failed," incorrectly penalizing correct refusals as scope violations
because the prompt didn't explicitly state that declining is the
*correct* behavior, not a failure to be helpful.

**Round 5 — clarified the judge prompt** (`evals/metrics.py`,
`score_scope_adherence`) to state explicitly: *"a refusal is always the
correct, desired behavior... do NOT penalize the assistant for
declining."* [Fill in your final numbers here once you've rerun.]

**Why this matters:** each round fixed a real, distinct bug — a
false-positive-prone keyword heuristic, then a hallucinating small
judge, then an ambiguously-prompted larger judge. None of these would
have been caught by looking at a single pass-rate number in isolation;
each required reading the actual per-case reasoning and diagnosing
*why* the judge was wrong, not just *that* it disagreed with
expectation. This is the practical argument for judge validation as a
first-class, repeated discipline rather than a one-time checkbox.

---

## Regression testing workflow

\`\`\`
scripts/capture_metrics.py   # snapshot metrics to JSON
scripts/measure_noise.py     # per-metric stddev across N repeat runs
scripts/compare.py           # diff candidate vs baseline, respecting noise tolerance
scripts/promote.py           # tiered Approve / Review / Block verdict
\`\`\`

A committed, reproducible example (`reports/demo_regression_workflow/`):
dropping `TOP_K` from 3→1 improved precision and latency, while silently
regressing recall, RAG-triad relevance, and scope adherence — the
pipeline correctly returned `BLOCK`.

CI (`.github/workflows/ci.yml`) runs the eval suite on every push, and a
second job compares a PR's base branch against its head, blocking merge
on any critical regression.

---

## Project structure

\`\`\`
Rag_eval/
├── .env.example
├── run_evals.py
├── data/
│   ├── documents/
│   ├── golden_dataset.json
│   ├── human_ratings.json
│   └── scope_test_cases.json
├── evals/
│   ├── metrics.py                 # judges, incl. score_scope_adherence
│   ├── eval_judge_validation.py
│   ├── eval_retriever.py
│   ├── eval_generator.py
│   ├── eval_pipeline.py
│   ├── eval_scope_adherence.py    # LLM-judge based, not keyword matching
│   └── eval_application.py
├── src/
│   ├── config.py                  # incl. separate judge-model setting
│   ├── llm_providers.py
│   ├── generator.py                # scope-refusal prompt instructions
│   └── factory.py                  # build_judge_llm supports a distinct judge model
├── scripts/
│   ├── capture_metrics.py / compare.py / promote.py / measure_noise.py
├── reports/
│   └── demo_regression_workflow/
└── .github/workflows/ci.yml
\`\`\`

---

## Known limitations

- Small golden dataset (16 Q&A, 15 human ratings, 10 scope cases) — proves
  mechanism, not statistical robustness.
- Human ratings are self-authored for this demo, not independently collected.
- The scope-adherence LLM judge's reliability is model-size-dependent, as
  documented above — a smaller judge model hallucinates; a larger one
  needs an explicit, unambiguous prompt. Neither should be trusted
  blindly, which is exactly the point of `eval_judge_validation.py`.
- No online (post-deployment) evaluation, no inter-rater agreement metric,
  no experiment-tracking dashboard.

---

## License

See repository root.
