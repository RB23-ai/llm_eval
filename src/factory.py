"""Single place that wires config -> providers -> index -> pipeline.

Every eval script and the demo script import `build_pipeline()` from here so
there is exactly one way the system gets assembled, no matter which provider
mode you're running in.
"""
from __future__ import annotations

import os
from functools import lru_cache

from src.config import SETTINGS, Settings
from src.embedding_providers import get_embedding_provider
from src.generator import Generator
from src.ingest import build_index
from src.llm_providers import get_llm_provider
from src.rag_pipeline import RagPipeline
from src.retriever import Retriever

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")

# Illustrative paid-API pricing (USD per 1K tokens) used only so the ops
# eval has something to report when LLM_PROVIDER is "openai"/"anthropic".
# Open-source local/offline providers cost $0 per token by construction.
_PAID_PRICING = {
    "openai": {"input": 0.00015, "output": 0.0006},       # gpt-4o-mini-ish
    "anthropic": {"input": 0.003, "output": 0.015},        # claude-sonnet-ish
}


@lru_cache(maxsize=None)
def build_pipeline(settings: Settings = SETTINGS) -> RagPipeline:
    # Cached: within one process (one `pytest` / `run_evals.py` run), every
    # eval suite asks for a pipeline built from the same settings. Rebuilding
    # from scratch each time is wasteful for HF/Ollama models and actively
    # breaks the Chroma backend, which deletes and recreates its on-disk
    # store on init -- concurrent rebuilds race on that file. One build,
    # reused everywhere, matches how a real deployed pipeline behaves too:
    # it's built once at startup, not per-request.
    embedder = get_embedding_provider(settings.embedding_provider, settings.hf_embedding_model)
    index = build_index(
        DOCUMENTS_DIR,
        embedder,
        settings.chunk_size_words,
        settings.chunk_overlap_words,
        vector_store=settings.vector_store,
    )
    retriever = Retriever(index=index, embedder=embedder)

    llm = get_llm_provider(settings.llm_provider, settings)
    generator = Generator(llm=llm)

    pricing = _PAID_PRICING.get(settings.llm_provider, {"input": 0.0, "output": 0.0})
    return RagPipeline(
        retriever=retriever,
        generator=generator,
        top_k=settings.top_k,
        cost_per_1k_input_tokens=pricing["input"],
        cost_per_1k_output_tokens=pricing["output"],
    )


@lru_cache(maxsize=None)
def build_judge_llm(settings: Settings = SETTINGS):
    if settings.judge_provider == "ollama":
        from src.llm_providers import OllamaLLM
        return OllamaLLM(settings.ollama_judge_model, settings.ollama_host)
    return get_llm_provider(settings.judge_provider, settings)