"""VectorStore protocol + a FAISS-backed implementation.

The protocol is deliberately small so a second backend (LanceDB, Qdrant,
pgvector, etc.) can be dropped in without touching pipeline.py. The store
owns both the vectors *and* the ChipSpec metadata, because every match needs
to return the spec it came from.

FaissStore serialization: a directory containing
    index.faiss    — the FAISS index (IndexFlatIP, cosine on unit vectors)
    specs.json     — JSON list of ChipSpec dicts, position = FAISS id
"""

from __future__ import annotations

import json
import os
from typing import List, Protocol, Tuple, runtime_checkable

import faiss
import numpy as np

from .chips import ChipSpec


@runtime_checkable
class VectorStore(Protocol):
    """Minimal interface every backend must satisfy."""

    embedding_dim: int

    def add(self, embeddings: np.ndarray, specs: List[ChipSpec]) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[float, ChipSpec]]: ...
    def save(self, path: str) -> None: ...
    def __len__(self) -> int: ...


class FaissStore:
    """IndexFlatIP cosine store with a JSON sidecar of ChipSpecs."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.specs: List[ChipSpec] = []

    def add(self, embeddings: np.ndarray, specs: List[ChipSpec]) -> None:
        if embeddings.shape[0] != len(specs):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) and specs ({len(specs)}) length mismatch"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding dim {embeddings.shape[1]} != store dim {self.embedding_dim}"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.specs.extend(specs)

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[float, ChipSpec]]:
        if len(self.specs) == 0:
            return []
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        k = min(top_k, len(self.specs))
        sims, idxs = self.index.search(query, k)
        out: List[Tuple[float, ChipSpec]] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if 0 <= idx < len(self.specs):
                out.append((float(sim), self.specs[idx]))
        return out

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "specs.json"), "w") as f:
            json.dump([s.to_dict() for s in self.specs], f)

    @classmethod
    def load(cls, path: str) -> "FaissStore":
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "specs.json"), "r") as f:
            spec_dicts = json.load(f)
        store = cls.__new__(cls)
        store.embedding_dim = index.d
        store.index = index
        store.specs = [ChipSpec.from_dict(d) for d in spec_dicts]
        return store

    def __len__(self) -> int:
        return len(self.specs)
