"""Scoring retrieved funds.

Changes from the notebook version:
  * NAV is no longer part of the score. A NAV of 900 vs 15 only reflects when a fund
    launched and its unit price, not how good it is.
  * Returns and costs are compared with peers in the same category ("beats 80% of Large
    Cap funds"), so liquid funds are not penalised for not matching small-cap returns and
    one outlier can't squash everyone else's score (as min-max scaling did).
  * The 1Y/3Y/5Y return weights follow the investor's time horizon when it can be read
    from the goal ("for 5 years", "6 months", "retirement").
  * Funds whose category doesn't suit that horizon, or that sit outside the requested
    risk band (fallback mode), are scaled down rather than silently mixed in.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .categories import profile_for

_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-|to)?\s*(?:\d+(?:\.\d+)?\s*)?(years?|yrs?|yr|y)\b")
_MONTHS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(?:\d+(?:\.\d+)?\s*)?(months?|mos?)\b")
_WEEKS = re.compile(r"(\d+(?:\.\d+)?)\s*(weeks?|wks?)\b")
_HORIZON_WORDS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(emergency|park(ing)?|few (days|weeks)|overnight)\b"), 0.25),
    (re.compile(r"\bshort[- ]?term\b"), 1.0),
    (re.compile(r"\b(medium|mid)[- ]?term\b"), 3.0),
    (re.compile(r"\b(long[- ]?term|retire(ment)?|pension|child(ren)?'?s? (education|future)|wealth creation)\b"), 7.0),
)


def parse_horizon_years(goal: str) -> float | None:
    """Best-effort investment horizon in years from free text, or None if not stated."""
    text = goal.lower()
    if m := _YEARS.search(text):
        return float(m.group(1))
    if m := _MONTHS.search(text):
        return float(m.group(1)) / 12
    if m := _WEEKS.search(text):
        return float(m.group(1)) / 52
    for pattern, years in _HORIZON_WORDS:
        if pattern.search(text):
            return years
    return None


def return_weights(horizon_years: float | None) -> tuple[float, float, float]:
    """Weights for (1Y, 3Y, 5Y) returns."""
    if horizon_years is None:
        return (0.25, 0.35, 0.40)
    if horizon_years <= 1:
        return (0.60, 0.30, 0.10)
    if horizon_years <= 3:
        return (0.25, 0.50, 0.25)
    return (0.10, 0.35, 0.55)


def horizon_fit(category: str, horizon_years: float | None) -> float:
    """1.0 if the category suits the horizon, falling towards 0 as it gets less suitable."""
    if horizon_years is None:
        return 1.0
    profile = profile_for(category)
    if horizon_years < profile.min_years:
        return max(0.0, horizon_years / profile.min_years) if profile.min_years else 1.0
    if profile.max_years is not None and horizon_years > profile.max_years:
        return math.sqrt(profile.max_years / horizon_years) if profile.max_years else 0.0
    return 1.0


@dataclass(frozen=True)
class RankingWeights:
    relevance: float = 0.40
    returns: float = 0.40
    cost: float = 0.20


@dataclass
class RankedFund:
    fund: dict
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


def rank_funds(
    candidates: list[Document],
    funds_by_code: dict[int, dict],
    goal: str,
    *,
    weights: RankingWeights | None = None,
    top_k: int = 5,
    risk_band: tuple[int, int] | None = None,
) -> list[RankedFund]:
    weights = weights or RankingWeights()
    horizon = parse_horizon_years(goal)
    w1, w3, w5 = return_weights(horizon)
    total_weight = (weights.relevance + weights.returns + weights.cost) or 1.0

    def raw_relevance(doc: Document) -> float:
        return 0.7 * doc.metadata.get("semantic_score", 0.0) + 0.3 * doc.metadata.get("keyword_score", 0.0)

    # Relative to the best match: absolute cosine levels depend on the embedding model,
    # but the *spread* is what tells "tax saving" -> ELSS apart from loosely related funds.
    best_relevance = max((raw_relevance(d) for d in candidates), default=0.0) or 1.0

    ranked: list[RankedFund] = []
    for doc in candidates:
        fund = funds_by_code.get(int(doc.metadata["scheme_code"]))
        if fund is None:
            continue
        relevance = raw_relevance(doc) / best_relevance
        returns = w1 * fund["returns_1yr_beats"] + w3 * fund["returns_3yr_beats"] + w5 * fund["returns_5yr_beats"]
        cost = fund["expense_beats"]
        base = (weights.relevance * relevance + weights.returns * returns + weights.cost * cost) / total_weight

        fit = horizon_fit(fund["category"], horizon)
        risk_fit = 1.0
        if risk_band is not None:
            lo, hi = risk_band
            distance = max(lo - fund["risk_step"], fund["risk_step"] - hi, 0)
            risk_fit = max(0.0, 1.0 - 0.15 * distance)
        score = base * (0.6 + 0.4 * fit) * risk_fit

        ranked.append(
            RankedFund(
                fund=fund,
                score=round(score, 4),
                breakdown={
                    "relevance": round(relevance, 3),
                    "returns": round(returns, 3),
                    "cost": round(cost, 3),
                    "horizon_fit": round(fit, 3),
                    "risk_fit": round(risk_fit, 3),
                },
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    # The dataset has a few schemes listed twice under different codes; show each name once.
    seen: set[str] = set()
    unique = []
    for item in ranked:
        name = item.fund["scheme_name"].lower()
        if name not in seen:
            seen.add(name)
            unique.append(item)
    return unique[:top_k]
