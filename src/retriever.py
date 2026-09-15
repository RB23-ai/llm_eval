"""The retriever component, evaluated in isolation in evals/eval_retriever.py."""
from __future__ import annotations

from dataclasses import dataclass

from src.embedding_providers import BaseEmbeddings
from src.ingest import BaseVectorIndex, Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, index: BaseVectorIndex, embedder: BaseEmbeddings):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed([query])[0]
        results = self.index.search(query_vec, top_k)
        return [RetrievedChunk(chunk=c, score=s) for c, s in results]
