"""Grounded, deterministic fund explanations.

Used three ways: as the fallback when the LLM fails or invents numbers, as the answer when
no LLM is configured, and as the training targets in scripts/train_lora.py (the old
targets said "steady returns ... long-term growth" for every fund, including liquid funds
and funds with negative returns).
"""

from __future__ import annotations

import math
import re

from .categories import profile_for, short_name, with_article
from .data import RISK_BANDS, RISK_LABELS

_LABELS = {"1yr": "1-year", "3yr": "3-year", "5yr": "5-year"}


def fmt(value: object, digits: int = 2) -> str:
    """Compact number formatting: 12.0 -> '12', 0.355 -> '0.36', NaN/None -> 'n/a'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def pct(share: float) -> int:
    return int(round(100 * float(share)))


def horizon_text(years: float | None) -> str:
    if years is None:
        return "not stated"
    if years < 1:
        months = max(1, round(years * 12))
        return f"about {months} month{'s' if months != 1 else ''}"
    return f"about {fmt(years, 1)} year{'s' if years != 1 else ''}"


def _return_order(horizon_years: float | None) -> list[str]:
    if horizon_years is not None and horizon_years <= 1:
        return ["1yr", "3yr"]
    if horizon_years is not None and horizon_years <= 3:
        return ["3yr", "1yr"]
    return ["5yr", "3yr"]


def peer_summary(fund: dict) -> str:
    peers = int(fund.get("peer_count") or 0)
    if peers == 0:
        return "No other funds in this category to compare with."
    parts = [
        f"{_LABELS[k]} return beats {pct(fund[f'returns_{k}_beats'])}%"
        for k in ("5yr", "3yr", "1yr")
    ]
    return (
        f"Compared with {peers} other {fund['category']}: " + ", ".join(parts)
        + f"; expense ratio is lower than {pct(fund['expense_beats'])}% of them."
    )


def _beats(share: float, first: bool, peers: int) -> str:
    whom = f"the {peers} other funds in its category" if first else "them"
    everyone = f"all {peers} other funds in its category" if first else "all of them"
    if pct(share) >= 100:
        return f"beats {everyone}"
    if pct(share) <= 0:
        return f"trails {everyone}"
    return f"beats {pct(share)}% of {whom}"


def _risk_step(fund: dict) -> int | None:
    step = fund.get("risk_step")
    if step is None or (isinstance(step, float) and math.isnan(step)):
        return None
    return int(step)


def risk_fit_text(fund: dict, requested_risk: str | None) -> str:
    """How the fund's riskometer level compares with the band the investor picked."""
    step = _risk_step(fund)
    if not requested_risk or requested_risk not in RISK_BANDS or step is None:
        return "no risk preference given"
    lo, hi = RISK_BANDS[requested_risk]
    band = f"which covers {RISK_LABELS[lo]} and {RISK_LABELS[hi]}"
    if step < lo:
        return f"lower than the {requested_risk} risk you asked for, {band}"
    if step > hi:
        return f"higher than the {requested_risk} risk you asked for, {band}"
    return f"matches the {requested_risk} risk you asked for"


def describe_fund(
    fund: dict, goal: str = "", horizon_years: float | None = None, requested_risk: str | None = None
) -> str:
    profile = profile_for(fund["category"])
    peers = int(fund.get("peer_count") or 0)
    kind = short_name(fund["category"])
    noun = kind if kind.endswith("funds") else f"{kind} fund"
    asset = "" if profile.asset_class in noun.replace("-", " ") else f" ({profile.asset_class})"
    sentences = [
        f"{fund['scheme_name']} is {with_article(noun)}{asset} rated {fund['risk_label']} risk on the riskometer."
    ]
    fit = risk_fit_text(fund, requested_risk)
    if fit.startswith(("lower", "higher")):
        sentences.append(f"That is {fit.split(', which')[0]}.")

    clauses = []
    for key in _return_order(horizon_years):
        value = fund.get(f"returns_{key}")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        clause = f"{_LABELS[key]} return of {fmt(value)}%"
        if peers:
            clause += " " + _beats(fund[f"returns_{key}_beats"], not clauses, peers)
        clauses.append(clause)
    if clauses:
        sentences.append("Its " + ", and its ".join(clauses) + ".")

    expense = fund.get("expense_ratio")
    if expense is not None and not (isinstance(expense, float) and math.isnan(expense)):
        if peers and pct(fund["expense_beats"]) >= 100:
            sentences.append(f"Its expense ratio of {fmt(expense)}% is the lowest in its category.")
        elif peers and pct(fund["expense_beats"]) <= 0:
            sentences.append(f"Its expense ratio of {fmt(expense)}% is the highest in its category.")
        elif peers:
            sentences.append(
                f"At {fmt(expense)}%, its expense ratio is lower than {pct(fund['expense_beats'])}% of peers."
            )
        else:
            sentences.append(f"Its expense ratio is {fmt(expense)}%.")

    if horizon_years is None:
        sentences.append(f"Funds in this category usually suit a holding period of {profile.horizon}.")
    else:
        from .ranking import horizon_fit  # local import avoids a cycle

        if horizon_fit(fund["category"], horizon_years) >= 0.999:
            sentences.append(
                f"Your horizon of {horizon_text(horizon_years)} fits this category's suggested "
                f"holding period of {profile.horizon}."
            )
        else:
            sentences.append(
                f"This category usually suits {profile.horizon}, so it may not match a horizon "
                f"of {horizon_text(horizon_years)}."
            )

    if profile.lock_in:
        sentences.append(f"Note the {profile.lock_in}.")
    r1 = fund.get("returns_1yr")
    if r1 is not None and not (isinstance(r1, float) and math.isnan(r1)) and r1 < 0:
        sentences.append(f"Its 1-year return is negative ({fmt(r1)}%), so expect short-term ups and downs.")
    return " ".join(sentences)


def prompt_inputs(
    fund: dict, goal: str, horizon_years: float | None, requested_risk: str | None = None
) -> dict[str, str]:
    """Variables for prompts.EXPLAIN_TEMPLATE (also used to build the training set)."""
    nav = fund.get("nav")
    return {
        "goal": goal.strip(),
        "requested_risk": requested_risk or "no preference",
        "risk_fit": risk_fit_text(fund, requested_risk),
        "scheme_name": fund["scheme_name"],
        "category": fund["category"],
        "risk_label": fund["risk_label"],
        "nav_text": "n/a" if nav is None or (isinstance(nav, float) and math.isnan(nav)) else f"₹{fmt(nav, 4)}",
        "returns_1yr": fmt(fund.get("returns_1yr")),
        "returns_3yr": fmt(fund.get("returns_3yr")),
        "returns_5yr": fmt(fund.get("returns_5yr")),
        "expense_ratio": fmt(fund.get("expense_ratio")),
        "peer_summary": peer_summary(fund),
        "horizon": horizon_text(horizon_years),
        "suggested_horizon": profile_for(fund["category"]).horizon,
    }


_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def cites_a_percentage(text: str) -> bool:
    return bool(_PERCENT.search(text))


def is_grounded(text: str, inputs: dict[str, str]) -> bool:
    """True if every percentage in ``text`` appears in the facts given to the model."""
    allowed = {float(n) for value in inputs.values() for n in _NUMBER.findall(str(value))}
    for match in _PERCENT.findall(text):
        number = float(match)
        if not any(abs(number - a) <= 0.051 for a in allowed):
            return False
    return True


# Qualitative claims a model can get wrong without citing a number ("low expense ratio" for a
# fund that is pricier than most of its peers), checked against the peer comparison.
_CHEAP = re.compile(
    r"\b(?:low|lower|lowest|cheap|cheaper|cheapest|competitive|attractive|reasonable|minimal|modest|economical)"
    r"\s+(?:expense|expenses|cost|costs|fee|fees|charges)\b"
    r"|\b(?:cost[- ]effective|cost[- ]efficient|inexpensive|low[- ]cost|budget[- ]friendly)\b"
    r"|\bexpense ratio\b(?:[^.!?]|\.(?=\d)){0,40}?\b(?:is|remains|looks|seems)\s+"
    r"(?:very\s+|quite\s+|relatively\s+|fairly\s+)?(?:low|competitive|reasonable|attractive|modest)\b",
    re.I,
)
_PRICEY = re.compile(
    r"\b(?:high|higher|highest|steep|hefty)\s+(?:expense|expenses|cost|costs|fee|fees|charges)\b"
    r"|\b(?:expensive|costly|pricey)\b",
    re.I,
)
_STRONG_RETURNS = re.compile(
    r"\b(?:strong|high|higher|impressive|excellent|superior|robust|solid|outstanding|stellar|exceptional|"
    r"above[- ]average|healthy|good|great|attractive)\s+(?:[\w-]+\s+){0,2}?(?:returns?|performance|track record)\b"
    r"|\bout-?perform(?:s|ed|ing|ance)?\b|\b(?:top|best)[- ]perform(?:ing|er|ers)\b",
    re.I,
)


def _number(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def contradictions(text: str, fund: dict) -> list[str]:
    """Claims in ``text`` that the fund's peer comparison says are false."""
    if not int(fund.get("peer_count") or 0):
        return []
    problems = []
    cost = _number(fund.get("expense_beats"))
    if cost is not None and _number(fund.get("expense_ratio")) is not None:
        if cost < 0.5 and _CHEAP.search(text):
            problems.append(f"calls the cost low, but the expense ratio is lower than only {pct(cost)}% of peers")
        if cost > 0.5 and _PRICEY.search(text):
            problems.append(f"calls the cost high, but the expense ratio is lower than {pct(cost)}% of peers")
    shares = [
        _number(fund.get(f"returns_{k}_beats"))
        for k in ("1yr", "3yr", "5yr")
        if _number(fund.get(f"returns_{k}")) is not None
    ]
    shares = [s for s in shares if s is not None]
    if shares and max(shares) < 0.5 and _STRONG_RETURNS.search(text):
        problems.append(f"praises the returns, but they beat at most {pct(max(shares))}% of peers in any period")
    return problems
