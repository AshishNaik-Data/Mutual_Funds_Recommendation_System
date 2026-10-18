import pytest
from langchain_core.documents import Document

from mf_recommender.ranking import horizon_fit, parse_horizon_years, rank_funds, return_weights


@pytest.mark.parametrize(
    "goal, years",
    [
        ("grow my money for 5 years", 5.0),
        ("over the next 10 yrs", 10.0),
        ("I need it in 6 months", 0.5),
        ("3-5 years", 3.0),
        ("park cash for 2 weeks", 2 / 52),
        ("emergency fund", 0.25),
        ("long-term wealth creation", 7.0),
        ("short term goal", 1.0),
        ("I like growth", None),
    ],
)
def test_parse_horizon_years(goal, years):
    assert parse_horizon_years(goal) == pytest.approx(years) if years is not None else parse_horizon_years(goal) is None


def test_return_weights_follow_horizon():
    assert return_weights(0.5)[0] > return_weights(0.5)[2]  # short: 1Y matters most
    assert return_weights(10)[2] > return_weights(10)[0]  # long: 5Y matters most
    for h in (None, 0.5, 2, 10):
        assert sum(return_weights(h)) == pytest.approx(1.0)


def test_horizon_fit():
    assert horizon_fit("Liquid Mutual Funds", 0.25) == 1.0
    assert horizon_fit("Liquid Mutual Funds", 10) < 0.5  # parking 10-year money in liquid funds
    assert horizon_fit("Small Cap Mutual Funds", 1) < 0.2
    assert horizon_fit("Large Cap Mutual Funds", None) == 1.0


def _doc(code, semantic=0.5, keyword=0.0):
    return Document(page_content="x", metadata={"scheme_code": code, "semantic_score": semantic, "keyword_score": keyword})


def _fund(code, name, **kw):
    base = dict(
        scheme_code=code, scheme_name=name, category="Large Cap Mutual Funds", risk_step=5,
        returns_1yr_beats=0.5, returns_3yr_beats=0.5, returns_5yr_beats=0.5, expense_beats=0.5, nav=10.0,
    )
    return {**base, **kw}


def test_nav_does_not_affect_score():
    funds = {1: _fund(1, "A", nav=10.0), 2: _fund(2, "B", nav=5000.0)}
    ranked = rank_funds([_doc(1), _doc(2)], funds, "growth")
    assert ranked[0].score == ranked[1].score


def test_better_peer_returns_and_lower_cost_rank_higher():
    funds = {1: _fund(1, "A"), 2: _fund(2, "B", returns_5yr_beats=0.9, expense_beats=0.9)}
    ranked = rank_funds([_doc(1), _doc(2)], funds, "growth for 10 years")
    assert [r.fund["scheme_code"] for r in ranked] == [2, 1]


def test_duplicate_scheme_names_are_shown_once():
    funds = {1: _fund(1, "Same Fund"), 2: _fund(2, "same fund", returns_5yr_beats=0.1), 3: _fund(3, "Other")}
    ranked = rank_funds([_doc(1), _doc(2), _doc(3)], funds, "growth", top_k=5)
    assert len(ranked) == 2


def test_out_of_band_risk_is_penalised():
    funds = {1: _fund(1, "A", risk_step=5), 2: _fund(2, "B", risk_step=1)}
    ranked = rank_funds([_doc(1), _doc(2)], funds, "growth", risk_band=(0, 1))
    assert ranked[0].fund["scheme_code"] == 2
    assert ranked[1].breakdown["risk_fit"] < 1.0


def test_single_candidate_does_not_divide_by_zero():
    ranked = rank_funds([_doc(1, semantic=0.0)], {1: _fund(1, "A")}, "growth")
    assert len(ranked) == 1 and 0 <= ranked[0].score <= 1
