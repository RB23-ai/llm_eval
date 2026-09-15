from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root explicitly, and let it override any stale
# `set VAR=...` values left over in the current shell session.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    # "mock" | "hf_local" | "ollama" | "openai" | "anthropic"
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    # "mock" | "hf_sentence_transformers"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "mock")
    ollama_judge_model: str = os.getenv("OLLAMA_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
    # Open-source model choices (swap freely for any HF/Ollama chat model)
    hf_generation_model: str = os.getenv("HF_GENERATION_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Paid fallbacks (only used if llm_provider is "openai" or "anthropic")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Judge can be configured separately from the generator, e.g. a bigger
    # model judging a cheaper model's answers. Defaults to llm_provider.
    judge_provider: str = os.getenv("JUDGE_PROVIDER", os.getenv("LLM_PROVIDER", "mock"))
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))

    # "memory" (default, zero deps) | "chroma" (real embedded vector DB)
    vector_store: str = os.getenv("VECTOR_STORE", "memory")

    # Retrieval / eval knobs
    top_k: int = int(os.getenv("TOP_K", "3"))
    chunk_size_words: int = int(os.getenv("CHUNK_SIZE_WORDS", "60"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "15"))

    # Regression gates used by run_evals.py / CI (see evals/report.py)
    min_retriever_recall_at_k: float = float(os.getenv("MIN_RETRIEVER_RECALL", "0.7"))
    min_faithfulness: float = float(os.getenv("MIN_FAITHFULNESS", "0.7"))
    min_answer_relevance: float = float(os.getenv("MIN_ANSWER_RELEVANCE", "0.7"))
    max_p95_latency_seconds: float = float(os.getenv("MAX_P95_LATENCY_SECONDS", "8.0"))
    # How far the configured judge may drift from human ratings before
    # evals/eval_judge_validation.py fails.
    max_judge_mae: float = float(os.getenv("MAX_JUDGE_MAE", "0.3"))

    # --- Scope adherence (see data/scope_test_cases.json) ---
    min_scope_adherence: float = float(os.getenv("MIN_SCOPE_ADHERENCE", "0.9"))

    # --- Operational evals (latency percentiles, reliability, cost) ---
    ops_repeat_runs: int = int(os.getenv("OPS_REPEAT_RUNS", "3"))
    ops_warmup_runs: int = int(os.getenv("OPS_WARMUP_RUNS", "1"))
    ops_daily_query_volume: int = int(os.getenv("OPS_DAILY_QUERY_VOLUME", "50000"))

    # --- Regression testing noise thresholds (scripts/measure_noise.py) ---
    noise_runs: int = int(os.getenv("NOISE_RUNS", "5"))
    noise_sigma_multiplier: float = float(os.getenv("NOISE_SIGMA_MULTIPLIER", "2.0"))


SETTINGS = Settings()