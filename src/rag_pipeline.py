"""The assembled RAG pipeline, evaluated as a whole in evals/eval_pipeline.py
and evals/eval_application.py (the RAG Triad + application-level checks)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.generator import Generator
from src.retriever import Retriever


@dataclass
class RagResult:
    question: str
    answer: str
    context_chunks: list[str]
    context_doc_ids: list[str]
    latency_seconds: float
    retrieval_latency_seconds: float
    generation_latency_seconds: float
    prompt_tokens_est: int
    completion_tokens_est: int
    cost_usd_est: float = 0.0
    input_cost_usd_est: float = 0.0
    output_cost_usd_est: float = 0.0


@dataclass
class RagPipeline:
    retriever: Retriever
    generator: Generator
    top_k: int = 3
    # rough per-1K-token cost used only to demonstrate ops evaluation;
    # 0.0 for mock/open-source local providers since there is no per-token bill
    cost_per_1k_input_tokens: float = field(default=0.0)
    cost_per_1k_output_tokens: float = field(default=0.0)

    def answer(self, question: str) -> RagResult:
        overall_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_elapsed = time.perf_counter() - retrieval_start

        context_texts = [r.chunk.text for r in retrieved]
        doc_ids = [r.chunk.doc_id for r in retrieved]

        generation_start = time.perf_counter()
        answer_text = self.generator.generate(question, context_texts)
        generation_elapsed = time.perf_counter() - generation_start

        elapsed = time.perf_counter() - overall_start

        prompt_tokens_est = _estimate_tokens(self.generator.build_prompt(question, context_texts))
        completion_tokens_est = _estimate_tokens(answer_text)
        input_cost = prompt_tokens_est / 1000 * self.cost_per_1k_input_tokens
        output_cost = completion_tokens_est / 1000 * self.cost_per_1k_output_tokens

        return RagResult(
            question=question,
            answer=answer_text,
            context_chunks=context_texts,
            context_doc_ids=doc_ids,
            latency_seconds=elapsed,
            retrieval_latency_seconds=retrieval_elapsed,
            generation_latency_seconds=generation_elapsed,
            prompt_tokens_est=prompt_tokens_est,
            completion_tokens_est=completion_tokens_est,
            cost_usd_est=input_cost + output_cost,
            input_cost_usd_est=input_cost,
            output_cost_usd_est=output_cost,
        )


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token is a standard rough estimate, good enough for ops dashboards
    return max(1, len(text) // 4)
