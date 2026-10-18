"""Embedding model factory."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i i'm im in into is it its me my of on or "
    "our so that the their them then there these this to us was we what when which who will "
    "with want would like looking need you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords (shared by BM25 and HashingEmbeddings)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (unigrams + bigrams, feature hashing).

    Lexical only, no model download. Used by the test-suite and as an offline fallback
    (``MF_EMBEDDING_MODEL=hashing``); use a sentence-transformer for real semantic search.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(model_name: str) -> Embeddings:
    if model_name == "hashing":
        return HashingEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings

    # Unit-length vectors so FAISS inner product == cosine similarity.
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
