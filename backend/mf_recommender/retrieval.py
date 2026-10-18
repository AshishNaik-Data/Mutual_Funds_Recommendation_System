"""Hybrid retriever: FAISS semantic search + BM25 keyword search, fused with RRF."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .embeddings import tokenize

MetadataFilter = Callable[[dict], bool]


def build_bm25(docs: list[Document]) -> BM25Retriever:
    return BM25Retriever.from_documents(docs, preprocess_func=tokenize, k=len(docs))


def make_filter(
    category: str | None = None, risk_band: tuple[int, int] | None = None
) -> MetadataFilter | None:
    if category is None and risk_band is None:
        return None

    def _filter(meta: dict) -> bool:
        if category is not None and meta.get("category") != category:
            return False
        if risk_band is not None and not (risk_band[0] <= meta.get("risk_step", -1) <= risk_band[1]):
            return False
        return True

    return _filter


class HybridFundRetriever(BaseRetriever):
    """Ranks funds by reciprocal rank fusion of FAISS and BM25 over one or more queries.

    Extra keyword arguments to ``invoke``:
      * ``metadata_filter`` - callable on Document.metadata; applied before ranking so a
        strict category/risk choice never gets crowded out by better-matching funds.
      * ``extra_queries`` - e.g. an LLM rewrite of the goal. Fusing the original *and*
        the rewrite means a poor rewrite can't drop good matches.
      * ``k`` - number of documents to return.

    Each returned Document carries ``semantic_score`` (cosine, 0-1), ``keyword_score``
    (BM25 relative to the best match for the query, 0-1) and ``retrieval_score`` (RRF).
    """

    vectorstore: FAISS
    bm25: BM25Retriever
    k: int = 30
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    rrf_k: int = 60

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _semantic(self, query: str, metadata_filter: MetadataFilter | None) -> list[tuple[Document, float]]:
        n = self.vectorstore.index.ntotal
        # The corpus is small (hundreds of funds), so score everything; with a filter,
        # fetch_k must cover the whole index or filtered-in funds could be missed.
        return self.vectorstore.similarity_search_with_score(query, k=n, fetch_k=n, filter=metadata_filter)

    def _keyword(self, query: str, metadata_filter: MetadataFilter | None) -> list[tuple[Document, float]]:
        tokens = self.bm25.preprocess_func(query)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.vectorizer.get_scores(tokens), dtype=float)
        best = scores.max() if scores.size else 0.0
        if best <= 0:
            return []
        out = []
        for i in np.argsort(-scores):
            if scores[i] <= 0:
                break  # no keyword overlap: no keyword vote
            doc = self.bm25.docs[i]
            if metadata_filter is None or metadata_filter(doc.metadata):
                out.append((doc, float(scores[i] / best)))
        return out

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        metadata_filter: MetadataFilter | None = None,
        extra_queries: Sequence[str] = (),
        k: int | None = None,
        **_: Any,
    ) -> list[Document]:
        queries = list(dict.fromkeys(q.strip() for q in (query, *extra_queries) if q and q.strip()))
        if not queries:
            return []
        k = k or self.k

        fused: dict[str, float] = {}
        semantic: dict[str, float] = {}
        keyword: dict[str, float] = {}
        docs: dict[str, Document] = {}

        for q in queries:
            for rank, (doc, score) in enumerate(self._semantic(q, metadata_filter), start=1):
                key = str(doc.metadata["scheme_code"])
                docs[key] = doc
                fused[key] = fused.get(key, 0.0) + self.semantic_weight / (self.rrf_k + rank)
                semantic[key] = max(semantic.get(key, 0.0), float(np.clip(score, 0.0, 1.0)))
            for rank, (doc, score) in enumerate(self._keyword(q, metadata_filter), start=1):
                key = str(doc.metadata["scheme_code"])
                docs[key] = doc
                fused[key] = fused.get(key, 0.0) + self.keyword_weight / (self.rrf_k + rank)
                keyword[key] = max(keyword.get(key, 0.0), score)

        best_possible = len(queries) * (self.semantic_weight + self.keyword_weight) / (self.rrf_k + 1)
        ranked = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [
            Document(
                id=key,
                page_content=docs[key].page_content,
                metadata={
                    **docs[key].metadata,
                    "retrieval_score": fused[key] / best_possible,
                    "semantic_score": semantic.get(key, 0.0),
                    "keyword_score": keyword.get(key, 0.0),
                },
            )
            for key in ranked
        ]
