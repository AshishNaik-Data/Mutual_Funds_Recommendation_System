import pytest
from langchain_core.language_models.fake import FakeListLLM

from mf_recommender.chains import (
    build_explain_chain,
    build_rewrite_chain,
    clean_generation,
    clean_rewrite,
    explanation_problem,
    is_degenerate,
    is_usable_explanation,
)
from mf_recommender.explanations import cites_a_percentage, describe_fund, is_grounded, prompt_inputs


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
        ("This fund has a low expense ratio and a 12.4% five-year return for its category.", True),
        ("This fund has a low expense ratio and strong long-term returns for its category.", False),  # no figures
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


# --- checks added after the first Phi-3 run (Banking and PSU, "short term", High risk) ---------


@pytest.fixture
def kotak(funds):
    """Expense ratio 0.37%: lower than only 13% of Banking and PSU peers."""
    return funds[funds["scheme_name"] == "Kotak Banking and PSU Debt Fund"].iloc[0].to_dict()


def test_gate_rejects_low_cost_claim_for_a_pricey_fund(kotak):
    facts = prompt_inputs(kotak, "I want a short term investment", 1.0, "High")
    # What untuned Phi-3 wrote for this fund, plus a real figure so only the claim is wrong.
    text = ("This could suit you given its 7.5% five-year return and low expense ratio compared to peers.")
    assert is_grounded(text, facts)
    assert "expense ratio is lower than only 13% of peers" in explanation_problem(text, facts, kotak)
    honest = "Its 5-year return of 7.5% beats 93% of peers, but at 0.37% its expense ratio is lower than 13% of peers."
    assert explanation_problem(honest, facts, kotak) is None


@pytest.mark.parametrize(
    "claim",
    ["a cost-effective choice", "its expense ratio of 0.37% is relatively low", "the lowest fees in its class"],
)
def test_cheap_claims_are_recognised(kotak, claim):
    facts = prompt_inputs(kotak, "", None)
    assert explanation_problem(f"With a 7.5% five-year return, it is {claim}.", facts, kotak)


def test_gate_rejects_praise_for_below_median_returns(funds):
    weak = funds[(funds[["returns_1yr_beats", "returns_3yr_beats", "returns_5yr_beats"]].max(axis=1) < 0.3)]
    fund = weak.iloc[0].to_dict()
    facts = prompt_inputs(fund, "", None)
    text = f"It has delivered strong returns, {facts['returns_5yr']}% over five years."
    assert "praises the returns" in explanation_problem(text, facts, fund)


def test_template_explanations_pass_the_gate(funds):
    for fund in funds.sample(80, random_state=0).to_dict("records"):
        for risk in (None, "Low", "High"):
            facts = prompt_inputs(fund, "grow my money for 5 years", 5.0, risk)
            text = describe_fund(fund, "grow my money for 5 years", 5.0, risk)
            if cites_a_percentage(text):
                assert explanation_problem(text, facts, fund) is None, text


def test_prompt_states_the_risk_the_investor_chose(kotak):
    from mf_recommender.prompts import EXPLAIN_PROMPT

    prompt = EXPLAIN_PROMPT.format(**prompt_inputs(kotak, "I want a short term investment", 1.0, "High"))
    assert "Risk I asked for: High" in prompt
    assert "Risk Level: Moderate (lower than the High risk you asked for, which covers High and Very High)" in prompt
    assert "with 'Moderate' risk" not in prompt
    assert "Risk I asked for: no preference" in EXPLAIN_PROMPT.format(**prompt_inputs(kotak, "x", None))


def test_template_mentions_a_risk_mismatch(kotak):
    assert "That is lower than the High risk you asked for." in describe_fund(kotak, "", 1.0, "High")
    assert "you asked for" not in describe_fund(kotak, "", 1.0, "Medium")
    assert "you asked for" not in describe_fund(kotak, "", 1.0, None)


def test_explain_chain_passes_risk_to_the_prompt(kotak):
    seen = []

    class Recorder(FakeListLLM):
        def _call(self, prompt, *args, **kwargs):
            seen.append(prompt)
            return super()._call(prompt, *args, **kwargs)

    out = build_explain_chain(Recorder(responses=["It suits you given its low expense ratio and 7.5% return."])).invoke(
        {"fund": kotak, "goal": "short term", "horizon_years": 1.0, "risk": "High"}
    )
    assert "Risk I asked for: High" in seen[0]
    assert out["source"] == "template"  # the low-cost claim is false for this fund
    assert "lower than the High risk you asked for" in out["text"]


def test_rewrite_prompt_steers_short_horizons_away_from_equity():
    from mf_recommender.prompts import REWRITE_PROMPT

    text = REWRITE_PROMPT.format(goal="I want a short term investment")
    assert "under 3 years means debt, liquid or money market funds, not equity" in text
    assert text.rstrip().endswith("Investor goal: I want a short term investment\nSearch query:")
