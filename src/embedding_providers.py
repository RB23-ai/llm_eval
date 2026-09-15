"""
Pluggable embedding providers.

MockEmbeddings requires no downloads and no internet access: it hashes words
into a fixed-size bag-of-words vector. It is not a good semantic embedding,
but it is deterministic and free, which makes it useful for CI and for
proving the evaluation *machinery* works before you plug in a real model.

HFSentenceTransformerEmbeddings wraps the open-source `sentence-transformers`
library (e.g. all-MiniLM-L6-v2), which is free, runs on CPU, and gives real
semantic similarity.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


class BaseEmbeddings(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix, L2-normalized rows."""


class MockEmbeddings(BaseEmbeddings):
    """Deterministic, offline, dependency-free hashed bag-of-words embedding."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])


class HFSentenceTransformerEmbeddings(BaseEmbeddings):
    """Real open-source embeddings via sentence-transformers (CPU-friendly)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # local import: optional dep

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)


def get_embedding_provider(provider: str, hf_model_name: str) -> BaseEmbeddings:
    if provider == "mock":
        return MockEmbeddings()
    if provider == "hf_sentence_transformers":
        return HFSentenceTransformerEmbeddings(hf_model_name)
    raise ValueError(f"Unknown embedding provider: {provider!r}")
