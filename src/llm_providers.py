"""
Pluggable LLM providers, all behind one interface: `generate(prompt) -> str`.

- MockLLM: offline, deterministic, extractive. No downloads or API keys.
  Used as the default so this whole project runs out of the box, and so the
  evaluation *harness* itself can be unit-tested without network access.
- HFLocalLLM: real open-source generation via Hugging Face `transformers`,
  e.g. Qwen2.5-1.5B-Instruct running on CPU/GPU locally. Free, no API key,
  but needs the model weights downloaded once.
- OllamaLLM: real open-source generation via a local Ollama server (e.g.
  `ollama run qwen2.5:1.5b`). Free, no API key, easiest way to run a real
  open-source chat model without wrestling with `transformers` directly.
- OpenAILLM / AnthropicLLM: paid fallbacks for teams that want frontier
  quality and are fine paying per token. Only used if explicitly configured.
"""
from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        ...


class MockLLM(BaseLLM):
    """
    A deterministic, offline stand-in for a real LLM.

    It does not "understand" anything — it extracts the sentence(s) from the
    supplied context that share the most words with the question. This is
    enough to make the whole pipeline (retrieval -> generation -> evaluation)
    runnable and testable with zero external dependencies, which is exactly
    what you want for CI and for a reviewer to `git clone && pytest` with no
    setup. Swap LLM_PROVIDER to "ollama" or "hf_local" for real generation.
    """

    name = "mock"

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        question, context = _parse_rag_prompt(prompt)
        if not context.strip():
            return "I don't have enough information in the provided context to answer that."
        sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
        q_words = set(_words(question))
        best_sentences = sorted(
            sentences,
            key=lambda s: len(q_words & set(_words(s))),
            reverse=True,
        )[:2]
        if not best_sentences or len(q_words & set(_words(best_sentences[0]))) == 0:
            return "I don't have enough information in the provided context to answer that."
        return ". ".join(best_sentences).strip() + "."


class HFLocalLLM(BaseLLM):
    """Real open-source generation via a local Hugging Face transformers model."""

    name = "hf_local"

    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # optional dep

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        output = self.model.generate(inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
        text = self.tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        return text.strip()


class OllamaLLM(BaseLLM):
    """Real open-source generation via a local Ollama server (free, no API key)."""

    name = "ollama"

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": max_tokens},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()


class OpenAILLM(BaseLLM):
    """Paid fallback: OpenAI chat completions."""

    name = "openai"

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ["OPENAI_API_KEY"]  # raises loudly if missing

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


class AnthropicLLM(BaseLLM):
    """Paid fallback: Anthropic messages API."""

    name = "anthropic"

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.environ["ANTHROPIC_API_KEY"]  # raises loudly if missing

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return "".join(block.get("text", "") for block in body.get("content", [])).strip()


def get_llm_provider(provider: str, settings) -> BaseLLM:
    if provider == "mock":
        return MockLLM()
    if provider == "hf_local":
        return HFLocalLLM(settings.hf_generation_model)
    if provider == "ollama":
        return OllamaLLM(settings.ollama_model, settings.ollama_host)
    if provider == "openai":
        return OpenAILLM(settings.openai_model)
    if provider == "anthropic":
        return AnthropicLLM(settings.anthropic_model)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def _words(text: str) -> list[str]:
    return [w.strip(".,?!:;\"'()").lower() for w in text.split() if w.strip(".,?!:;\"'()")]


def _parse_rag_prompt(prompt: str) -> tuple[str, str]:
    """Pulls the question and context back out of the standard RAG prompt template.

    The template (see src/generator.py) is:
        Context:
        {context}

        Question: {question}

        Answer:

    Context must be isolated to the text strictly between "Context:" and the
    *next* "Question:" marker, and question strictly between "Question:" and
    "Answer:" -- otherwise the parsed context ends up containing the literal
    "Question: ..." line, which lets an adversarial question get echoed back
    as if it were retrieved context. (This was a real bug caught by
    evals/eval_scope_adherence.py: adversarial prompts were leaking straight
    through because of it.)
    """
    question, context = "", ""
    if "Context:" in prompt and "Question:" in prompt:
        context = prompt.split("Context:", 1)[1].split("Question:")[0].strip()
        question = prompt.split("Question:", 1)[1].split("Answer:")[0].strip()
    return question, context
