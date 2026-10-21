import numpy as np
import pytest

from mf_recommender.categories import CATEGORY_PROFILES, short_name
from mf_recommender.data import RISK_LABELS, normalize_risk_choice


def test_every_dataset_category_has_a_profile(funds):
    assert set(funds["category"]) <= set(CATEGORY_PROFILES)


def test_risk_levels_map_to_riskometer_labels(funds):
    assert set(funds["risk_step"]) <= set(RISK_LABELS)
    overnight = funds[funds["category"] == "Overnight Mutual Funds"]
    small_cap = funds[funds["category"] == "Small Cap Mutual Funds"]
    assert set(overnight["risk_label"]) == {"Low"}
    assert set(small_cap["risk_label"]) == {"Very High"}


def test_zero_nav_is_treated_as_missing(funds):
    import pandas as pd

    from mf_recommender.config import Settings

    raw = pd.read_csv(Settings.from_env().data_path)
    failed_fetches = (pd.to_numeric(raw["NAV"], errors="coerce").fillna(0) <= 0).sum()  # saved as 0.0 originally
    assert not (funds["nav"] <= 0).any()
    assert funds["nav"].isna().sum() == failed_fetches


def test_peer_shares_are_bounded_and_best_fund_scores_one(funds):
    for col in ("returns_1yr_beats", "returns_3yr_beats", "returns_5yr_beats", "expense_beats"):
        assert funds[col].between(0, 1).all()
    large = funds[funds["category"] == "Large Cap Mutual Funds"]
    best = large.loc[large["returns_5yr"].idxmax()]
    cheapest = large.loc[large["expense_ratio"].idxmin()]
    assert best["returns_5yr_beats"] == 1.0
    assert cheapest["expense_beats"] == 1.0


@pytest.mark.parametrize("raw, expected", [("low", "Low"), ("MEDIUM", "Medium"), (" High ", "High"), ("", None), (None, None), ("any", None)])
def test_normalize_risk_choice(raw, expected):
    assert normalize_risk_choice(raw) == expected


def test_normalize_risk_choice_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_risk_choice("extreme")


def test_documents_carry_filter_metadata(docs, funds):
    assert len(docs) == len(funds)
    doc = docs[0]
    assert {"scheme_code", "scheme_name", "category", "risk_step"} <= set(doc.metadata)
    assert doc.metadata["category"] in doc.page_content
    assert not np.isnan(doc.metadata["risk_step"])


def test_short_names():
    assert short_name("Large Cap Mutual Funds") == "Large Cap"
    assert short_name("Index Funds") == "Index"
    assert short_name("Dynamic Bond") == "Dynamic Bond"
    assert short_name("Childrens Funds") == "children's"
