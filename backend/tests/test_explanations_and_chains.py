import pytest
from langchain_core.language_models.fake import FakeListLLM

from mf_recommender.chains import (
    build_explain_chain,
    build_rewrite_chain,
    clean_generation,
    clean_rewrite,
    is_degenerate,
    is_usable_explanation,
)
from mf_recommender.explanations import describe_fund, is_grounded, prompt_inputs


@pytest.fixture
def fund(funds):
    return funds[funds["category"] == "ELSS Mutual Funds"].iloc[0].to_dict()


def test_describe_fund_is_grammatical_and_grounded(fund):
    text = describe_fund(fund, "save tax for 5 years", 5.0)
    assert "Mutual Funds fund" not in text
    assert "an ELSS fund" in text
    assert "3-year lock-in" in text
    assert is_grounded(text, prompt_inputs(fund, "save tax for 5 years", 5.0))


def test_describe_fund_flags_horizon_mismatch(funds):
    liquid = funds[funds["category"] == "Liquid Mutual Funds"].iloc[0].to_dict()
    assert "may not match" in describe_fund(liquid, "", 10.0)
    assert "fits" in describe_fund(liquid, "", 0.25)


def test_describe_fund_warns_on_negative_returns(funds):
    neg = funds[funds["returns_1yr"] < 0].iloc[0].to_dict()
    assert "negative" in describe_fund(neg)


def test_is_grounded_catches_invented_numbers(fund):
    inputs = prompt_inputs(fund, "goal", None)
    assert is_grounded(f"It returned {inputs['returns_5yr']}% over five years.", inputs)
    assert not is_grounded("It returned 99.9% last year.", inputs)


def test_clean_generation_stops_at_markers():
    raw = " This fund suits you. It is cheap.\n\nUser goal: something else entirely"
    assert clean_generation(raw) == "This fund suits you. It is cheap."


def test_clean_rewrite():
    assert clean_rewrite(' Search query: "large cap equity fund, 10 years"\nmore') == "large cap equity fund, 10 years"
    assert clean_rewrite("of of of of of of of of of of") == ""
    assert clean_rewrite("::::::::::::::::::::") == ""
    assert clean_rewrite("") == ""


def test_is_degenerate():
    assert is_degenerate("of of of of of of of of of of")
    assert not is_degenerate("This fund has a low expense ratio and strong five-year returns.")


@pytest.mark.parametrize(
    "text, ok",
    [
        ("This fund has a low expense ratio and strong long-term returns for its category.", True),
        ("ofmatofargeofrimarimarimarimariofmatofargeofrimarimarimarimari", False),  # run-together junk
        ("Good fund.", False),  # too short
        ("This fund returned 87.3% every single year, guaranteed by the fund house.", False),  # invented number
    ],
)
def test_is_usable_explanation(text, ok):
    assert is_usable_explanation(text, {"facts": "5Y 12.4%"}) is ok


def test_explain_chain_rejects_junk_output(fund):
    junk = "ofmatofargeofrimarimarimarimariofmatofargeofrimarimarimarimari"
    assert build_explain_chain(FakeListLLM(responses=[junk])).invoke(_explain_input(fund))["source"] == "template"


def _explain_input(fund):
    return {"fund": fund, "goal": "save tax for 5 years", "horizon_years": 5.0}


def test_explain_chain_uses_llm_answer_when_grounded(fund):
    answer = f"This ELSS fund has a {prompt_inputs(fund, '', None)['returns_5yr']}% five-year return and a lock-in."
    out = build_explain_chain(FakeListLLM(responses=[answer])).invoke(_explain_input(fund))
    assert out == {"text": answer, "source": "llm"}


def test_explain_chain_falls_back_on_hallucinated_numbers(fund):
    out = build_explain_chain(FakeListLLM(responses=["This fund returned 87.3% every single year, guaranteed."])).invoke(
        _explain_input(fund)
    )
    assert out["source"] == "template"
    assert "87.3" not in out["text"]


def test_explain_chain_falls_back_when_llm_raises(fund):
    class Broken(FakeListLLM):
        def _call(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

    out = build_explain_chain(Broken(responses=["x"])).invoke(_explain_input(fund))
    assert out["source"] == "template"


def test_explain_chain_batch(funds):
    rows = [r for r in funds.head(3).to_dict("records")]
    outs = build_explain_chain(None).batch([{"fund": r, "goal": "g", "horizon_years": None} for r in rows])
    assert [o["source"] for o in outs] == ["template"] * 3


def test_rewrite_chain():
    chain = build_rewrite_chain(FakeListLLM(responses=[" large cap equity fund for long-term growth\nUser goal: x"]))
    assert chain.invoke("I want my money to grow") == "large cap equity fund for long-term growth"
    assert build_rewrite_chain(None) is None
