from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake import FakeListLLM

from mf_recommender.api import create_app
from mf_recommender.nav import NavClient, NavQuote
from mf_recommender.service import UnknownCategoryError, nav_age_days

# Keys the existing React frontend reads.
LEGACY_KEYS = {"scheme_name", "NAV", "returns_1yr", "returns_3yr", "returns_5yr", "expense_ratio", "risk_level", "explanation"}


def test_recommend_respects_category_and_risk(service):
    result = service.recommend("long-term growth for 10 years", "Large Cap Mutual Funds", "High")
    assert len(result.funds) == 5
    assert result.notice is None
    for fund in result.funds:
        assert fund["category"] == "Large Cap Mutual Funds"
        assert fund["risk_level"] in {"High", "Very High"}
        assert LEGACY_KEYS <= set(fund)
    scores = [f["score"] for f in result.funds]
    assert scores == sorted(scores, reverse=True)


def test_impossible_combination_falls_back_with_notice(service):
    result = service.recommend("aggressive growth", "Small Cap Mutual Funds", "Low")
    assert result.funds
    assert result.notice and "Small Cap" in result.notice
    assert all(f["category"] == "Small Cap Mutual Funds" for f in result.funds)


def test_any_category_uses_the_goal(service):
    result = service.recommend("park my emergency money for 3 months", None, "Low")
    assert {f["category"] for f in result.funds} & {"Liquid Mutual Funds", "Overnight Mutual Funds", "Money Market Funds"}
    assert result.query["horizon_years"] == pytest.approx(0.25)


def test_category_is_case_insensitive_and_validated(service):
    assert service.resolve_category("large cap mutual funds") == "Large Cap Mutual Funds"
    assert service.resolve_category("Any") is None
    with pytest.raises(UnknownCategoryError):
        service.resolve_category("Crypto Funds")


def test_llm_rewrite_is_fused_with_original_goal(make_service):
    svc = make_service(llm=FakeListLLM(responses=["tax saving ELSS equity fund with lock-in"]))
    result = svc.recommend("I want to grow money for 5 years", None, "High", explain=False)
    assert result.query["rewritten"] == "tax saving ELSS equity fund with lock-in"
    assert any(f["category"] == "ELSS Mutual Funds" for f in result.funds)


def _fixed_nav(nav, nav_date):
    class FakeNav(NavClient):
        def latest(self, scheme_code):
            return NavQuote(nav=nav, date=nav_date)

    return FakeNav()


def test_live_nav_overrides_dataset_nav(make_service):
    today = date.today().isoformat()
    result = make_service(nav_client=_fixed_nav(123.45, today)).recommend("long-term growth", "Index Funds", "High")
    assert all(f["NAV"] == 123.45 and f["nav_source"] == "live" and f["nav_date"] == today for f in result.funds)


def test_old_nav_is_flagged_stale_not_live(make_service):
    # Kotak Banking and PSU's scheme code returned a NAV dated 2013-08-13, shown as "live".
    result = make_service(nav_client=_fixed_nav(10.78, "2013-08-13")).recommend("short term", "Index Funds", "High")
    assert {f["nav_source"] for f in result.funds} == {"stale"}
    week_old = (date.today() - timedelta(days=6)).isoformat()  # e.g. across a long holiday
    result = make_service(nav_client=_fixed_nav(10.78, week_old)).recommend("short term", "Index Funds", "High")
    assert {f["nav_source"] for f in result.funds} == {"live"}


def test_nav_age_days():
    assert nav_age_days("2026-10-10", date(2026, 10, 20)) == 10
    assert nav_age_days(None) is None and nav_age_days("not a date") is None


def test_nav_client_parses_and_caches():
    calls = []

    class Resp:
        status_code = 200

        def json(self):
            return {"data": [{"date": "16-10-2026", "nav": "45.6789"}]}

    class Session:
        def get(self, url, timeout):
            calls.append(url)
            return Resp()

    client = NavClient(session=Session())
    assert client.latest(100033) == NavQuote(45.6789, "2026-10-16")
    client.latest(100033)
    assert len(calls) == 1


def test_nav_client_failure_returns_none():
    import requests

    class Session:
        def get(self, url, timeout):
            raise requests.ConnectionError("offline")

    assert NavClient(session=Session()).latest(1) is None


@pytest.fixture
def client(service):
    with TestClient(create_app(service=service)) as c:
        yield c


def test_api_health_and_categories(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert {"llm_backend", "base_model", "adapter", "fine_tuned"} <= set(health)
    body = client.get("/categories").json()
    assert "Large Cap Mutual Funds" in body["categories"]
    assert body["risk_levels"] == ["Low", "Medium", "High"]


def test_api_recommend(client):
    resp = client.post("/recommend", json={"goal": "Steady income for 2 years", "category": "Corporate Bond Mutual Funds", "risk": "low"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["funds"]) == 5
    assert LEGACY_KEYS <= set(body["funds"][0])
    assert body["filters"] == {"category": "Corporate Bond Mutual Funds", "risk": "Low", "risk_band": [0, 1]}


def test_api_any_category_and_top_k(client):
    resp = client.post("/recommend", json={"goal": "Save tax for 5 years", "category": "Any", "risk": None, "top_k": 3})
    assert resp.status_code == 200 and len(resp.json()["funds"]) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"goal": "hi", "category": "Index Funds", "risk": "Low"},
        {"goal": "Long-term growth please", "category": "Crypto", "risk": "Low"},
        {"goal": "Long-term growth please", "category": "Index Funds", "risk": "Extreme"},
        {"goal": "Long-term growth please", "top_k": 50},
    ],
)
def test_api_rejects_bad_input(client, payload):
    assert client.post("/recommend", json=payload).status_code == 422


def test_api_cors(client):
    resp = client.options(
        "/recommend",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type,ngrok-skip-browser-warning"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] in {"*", "http://localhost:3000"}
