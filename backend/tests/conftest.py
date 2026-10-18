from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

# Make `mf_recommender` importable even when pytest is started from another folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mf_recommender.config import Settings  # noqa: E402
from mf_recommender.data import fund_documents, load_funds  # noqa: E402
from mf_recommender.embeddings import HashingEmbeddings  # noqa: E402
from mf_recommender.retrieval import HybridFundRetriever, build_bm25  # noqa: E402
from mf_recommender.service import RecommenderService  # noqa: E402
from mf_recommender.vectorstore import build_vectorstore  # noqa: E402


@pytest.fixture(scope="session")
def funds():
    return load_funds(Settings.from_env().data_path)  # honours MF_DATA_PATH


@pytest.fixture(scope="session")
def docs(funds):
    return fund_documents(funds)


@pytest.fixture(scope="session")
def retriever(docs):
    store = build_vectorstore(docs, HashingEmbeddings())
    return HybridFundRetriever(vectorstore=store, bm25=build_bm25(docs), k=30)


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return Settings(
        embedding_model="hashing",
        llm_backend="none",
        live_nav=False,
        index_dir=tmp_path_factory.mktemp("index") / "faiss",
    )


@pytest.fixture(scope="session")
def service(settings, funds, retriever):
    return RecommenderService(settings, funds, retriever, llm=None, nav_client=None)


@pytest.fixture
def make_service(settings, funds, retriever):
    def _make(llm=None, nav_client=None, **overrides):
        return RecommenderService(dataclasses.replace(settings, **overrides), funds, retriever, llm, nav_client)

    return _make
