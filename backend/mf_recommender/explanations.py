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


def describe_fund(fund: dict, goal: str = "", horizon_years: float | None = None) -> str:
    profile = profile_for(fund["category"])
    peers = int(fund.get("peer_count") or 0)
    kind = short_name(fund["category"])
    noun = kind if kind.endswith("funds") else f"{kind} fund"
    asset = "" if profile.asset_class in noun.replace("-", " ") else f" ({profile.asset_class})"
    sentences = [
        f"{fund['scheme_name']} is {with_article(noun)}{asset} rated {fund['risk_label']} risk on the riskometer."
    ]

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


def prompt_inputs(fund: dict, goal: str, horizon_years: float | None) -> dict[str, str]:
    """Variables for prompts.EXPLAIN_TEMPLATE (also used to build the training set)."""
    nav = fund.get("nav")
    return {
        "goal": goal.strip(),
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


def is_grounded(text: str, inputs: dict[str, str]) -> bool:
    """True if every percentage in ``text`` appears in the facts given to the model."""
    allowed = {float(n) for value in inputs.values() for n in _NUMBER.findall(str(value))}
    for match in _PERCENT.findall(text):
        number = float(match)
        if not any(abs(number - a) <= 0.051 for a in allowed):
            return False
    return True
