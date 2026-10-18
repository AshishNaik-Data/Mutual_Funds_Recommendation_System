from mf_recommender.data import RISK_BANDS
from mf_recommender.retrieval import make_filter
from mf_recommender.vectorstore import load_or_build_vectorstore
from mf_recommender.embeddings import HashingEmbeddings


def test_tax_saving_goal_finds_elss(retriever):
    docs = retriever.invoke("I want to save tax under 80C", k=5)
    assert docs and all(d.metadata["category"] == "ELSS Mutual Funds" for d in docs)


def test_emergency_goal_finds_short_term_debt(retriever):
    docs = retriever.invoke("park my emergency cash for a few weeks", k=5)
    assert {d.metadata["category"] for d in docs} <= {"Overnight Mutual Funds", "Liquid Mutual Funds"}


def test_category_and_risk_filters_are_respected(retriever):
    flt = make_filter("Corporate Bond Mutual Funds", RISK_BANDS["Low"])
    docs = retriever.invoke("steady income", metadata_filter=flt, k=50)
    assert docs
    for d in docs:
        assert d.metadata["category"] == "Corporate Bond Mutual Funds"
        assert 0 <= d.metadata["risk_step"] <= 1


def test_impossible_filter_returns_nothing(retriever):
    flt = make_filter("Small Cap Mutual Funds", RISK_BANDS["Low"])
    assert retriever.invoke("growth", metadata_filter=flt) == []


def test_scores_are_attached_and_sorted(retriever):
    docs = retriever.invoke("banking sector", k=10)
    scores = [d.metadata["retrieval_score"] for d in docs]
    assert scores == sorted(scores, reverse=True)
    for d in docs:
        assert 0 <= d.metadata["semantic_score"] <= 1
        assert 0 <= d.metadata["keyword_score"] <= 1


def test_extra_queries_are_fused(retriever):
    plain = retriever.invoke("good returns", k=5)
    fused = retriever.invoke("good returns", extra_queries=["funds that also hold gold"], k=5)
    assert [d.id for d in plain] != [d.id for d in fused]
    assert all("gold" in d.page_content.lower() for d in fused)


def test_index_is_reused_and_rebuilt_when_data_changes(docs, tmp_path):
    emb = HashingEmbeddings()
    first = load_or_build_vectorstore(docs, emb, tmp_path, "hashing")
    mtime = (tmp_path / "index.faiss").stat().st_mtime_ns
    again = load_or_build_vectorstore(docs, emb, tmp_path, "hashing")
    assert (tmp_path / "index.faiss").stat().st_mtime_ns == mtime  # loaded, not rebuilt
    assert again.index.ntotal == first.index.ntotal == len(docs)
    smaller = load_or_build_vectorstore(docs[:10], emb, tmp_path, "hashing")
    assert smaller.index.ntotal == 10
