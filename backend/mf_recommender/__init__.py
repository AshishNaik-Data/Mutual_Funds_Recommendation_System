"""LangChain-powered mutual fund recommender.

Pipeline: goal -> (optional LLM query rewrite) -> hybrid retrieval (FAISS + BM25,
fused with reciprocal rank fusion, filtered by category and risk) -> peer-relative
ranking -> live NAV -> grounded explanations (LLM with a deterministic fallback).
"""

from .config import Settings

__all__ = ["Settings"]
__version__ = "2.0.0"
