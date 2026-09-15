"""The generator component, evaluated in isolation in evals/eval_generator.py."""
from __future__ import annotations

from src.llm_providers import BaseLLM

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about LLM evaluation.
Answer ONLY using the information in the context below. If the context does
not contain the answer, say you don't have enough information.

You are strictly scoped to LLM-evaluation topics. If the question asks for
anything outside that scope (poems, creative writing, financial/investment
advice, travel planning, code unrelated to this topic, weather, etc.),
politely decline that part and explain you can only help with LLM
evaluation topics. If a question mixes an in-scope part with an
out-of-scope part, answer the in-scope part and decline the rest. Be
concise.

Context:
{context}

Question: {question}

Answer:"""


class Generator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def build_prompt(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, question: str, context_chunks: list[str], max_tokens: int = 300) -> str:
        prompt = self.build_prompt(question, context_chunks)
        return self.llm.generate(prompt, max_tokens=max_tokens)
