"""
Load raw .txt documents, chunk them, and build a searchable vector index.

Two vector store backends are supported behind the same `search()` interface
(`VECTOR_STORE` in src/config.py):

  "memory" (default) - a plain numpy matrix + cosine similarity. Zero extra
                        dependencies, instant startup, fine up to a few
                        thousand chunks. Good for CI and this demo's 8 docs.
  "chroma"            - a real embedded vector database (Chroma, via
                        `pip install chromadb`), persisted to disk under
                        .chroma/. This is what you'd actually reach for once
                        a corpus grows past a demo, and it's what lets you
                        say "I used a real vector DB" rather than "I hacked
                        together a numpy array" in an interview.

Both backends store the exact same chunks and are evaluated by the exact
same eval suite -- switching VECTOR_STORE=chroma changes zero lines in
evals/.
"""
from __future__ import annotations

import glob
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.embedding_providers import BaseEmbeddings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def load_documents(documents_dir: str) -> dict[str, str]:
    docs = {}
    for path in sorted(glob.glob(os.path.join(documents_dir, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    return docs


def chunk_text(doc_id: str, text: str, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size_words - overlap_words)
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + chunk_size_words]
        if not piece:
            continue
        chunks.append(Chunk(chunk_id=f"{doc_id}::chunk{i}", doc_id=doc_id, text=" ".join(piece)))
        if start + chunk_size_words >= len(words):
            break
    return chunks


class BaseVectorIndex(ABC):
    chunks: list[Chunk]

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Return the top_k (Chunk, similarity_score) pairs for a query vector."""


@dataclass
class InMemoryVectorIndex(BaseVectorIndex):
    chunks: list[Chunk]
    vectors: np.ndarray  # (n, d), L2-normalized

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ query_vector  # cosine sim, since rows are normalized
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


class ChromaVectorIndex(BaseVectorIndex):
    """
    Real embedded vector database backend, using Chroma's persistent local
    client (no separate server process required). Embeddings are computed
    up front with whichever `embedder` was configured and inserted directly,
    so Chroma is used purely as the ANN index + storage layer -- the
    embedding provider abstraction in src/embedding_providers.py is
    unchanged either way.
    """

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, persist_dir: str, collection_name: str):
        import chromadb

        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)  # rebuild fresh each run so the demo is reproducible
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        if chunks:
            self.collection.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors.tolist(),
                documents=[c.text for c in chunks],
                metadatas=[{"doc_id": c.doc_id} for c in chunks],
            )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        result = self.collection.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
        ids = result["ids"][0]
        distances = result["distances"][0]  # cosine distance = 1 - cosine similarity
        return [(self._by_id[cid], 1.0 - dist) for cid, dist in zip(ids, distances)]


def build_index(
    documents_dir: str,
    embedder: BaseEmbeddings,
    chunk_size_words: int,
    overlap_words: int,
    vector_store: str = "memory",
    persist_dir: str = ".chroma",
    collection_name: str = "rag_eval_showcase",
) -> BaseVectorIndex:
    docs = load_documents(documents_dir)
    all_chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        all_chunks.extend(chunk_text(doc_id, text, chunk_size_words, overlap_words))
    vectors = embedder.embed([c.text for c in all_chunks])

    if vector_store == "memory":
        return InMemoryVectorIndex(chunks=all_chunks, vectors=vectors)
    if vector_store == "chroma":
        return ChromaVectorIndex(all_chunks, vectors, persist_dir, collection_name)
    raise ValueError(f"Unknown vector store: {vector_store!r}")
