"""Orchestrates retrieval, ranking, live NAV and explanations for one request."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from .chains import build_explain_chain, build_rewrite_chain
from .config import Settings
from .data import RISK_BANDS, RISK_LABELS, clean_number, fund_documents, load_funds, normalize_risk_choice
from .embeddings import build_embeddings
from .nav import NavClient
from .ranking import RankingWeights, parse_horizon_years, rank_funds
from .retrieval import HybridFundRetriever, build_bm25, make_filter
from .vectorstore import load_or_build_vectorstore

log = logging.getLogger(__name__)

_UNSET: Any = object()


class UnknownCategoryError(ValueError):
    pass


def nav_age_days(nav_date: str | None, today: date | None = None) -> int | None:
    try:
        return ((today or date.today()) - date.fromisoformat(nav_date)).days if nav_date else None
    except ValueError:
        return None


@dataclass
class Recommendation:
    funds: list[dict]
    query: dict[str, Any]
    notice: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)


class RecommenderService:
    def __init__(
        self,
        settings: Settings,
        funds: pd.DataFrame,
        retriever: HybridFundRetriever,
        llm: Runnable | None = None,
        nav_client: NavClient | None = None,
    ):
        self.settings = settings
        self.funds = funds
        self.funds_by_code: dict[int, dict] = {int(r["scheme_code"]): r for r in funds.to_dict("records")}
        self.retriever = retriever
        self.llm = llm
        self.nav_client = nav_client
        self.rewrite_chain = build_rewrite_chain(llm) if settings.rewrite_query else None
        self.explain_chain = build_explain_chain(llm)
        self.weights = RankingWeights(settings.weight_relevance, settings.weight_returns, settings.weight_cost)
        self.categories: list[str] = sorted(funds["category"].unique())
        self._llm_lock = threading.Lock()  # one generation at a time on a single GPU
        self._rewrite_cached = lru_cache(maxsize=512)(self._rewrite_uncached)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embeddings: Embeddings | None = None,
        llm: Runnable | None = _UNSET,
        nav_client: NavClient | None = _UNSET,
    ) -> RecommenderService:
        funds = load_funds(settings.data_path)
        docs = fund_documents(funds)
        store = load_or_build_vectorstore(
            docs, embeddings or build_embeddings(settings.embedding_model), settings.index_dir, settings.embedding_model
        )
        retriever = HybridFundRetriever(vectorstore=store, bm25=build_bm25(docs), k=settings.candidate_k)
        if llm is _UNSET:
            from .llm import build_llm

            llm = build_llm(settings)
        if nav_client is _UNSET:
            nav_client = (
                NavClient(ttl_seconds=settings.nav_ttl_seconds, timeout=settings.nav_timeout_seconds)
                if settings.live_nav
                else None
            )
        return cls(settings, funds, retriever, llm, nav_client)

    # ------------------------------------------------------------------ helpers
    def resolve_category(self, category: str | None) -> str | None:
        if category is None or not category.strip() or category.strip().lower() in {"any", "all"}:
            return None
        lookup = {c.lower(): c for c in self.categories}
        try:
            return lookup[category.strip().lower()]
        except KeyError:
            raise UnknownCategoryError(f"Unknown category {category!r}") from None

    def _rewrite_uncached(self, goal: str) -> str:
        if self.rewrite_chain is None:
            return ""
        try:
            with self._llm_lock:
                rewritten = self.rewrite_chain.invoke(goal)
        except Exception:
            log.exception("Query rewrite failed; using the original goal only")
            return ""
        return "" if rewritten.strip().lower() == goal.strip().lower() else rewritten

    def _explain(
        self, ranked_funds: list[dict], goal: str, horizon: float | None, risk: str | None = None
    ) -> list[dict[str, str]]:
        inputs = [{"fund": f, "goal": goal, "horizon_years": horizon, "risk": risk} for f in ranked_funds]
        if not inputs:
            return []
        if self.llm is None:
            return self.explain_chain.batch(inputs)
        with self._llm_lock:
            return self.explain_chain.batch(inputs)

    # ------------------------------------------------------------------ main entry point
    def recommend(
        self,
        goal: str,
        category: str | None = None,
        risk: str | None = None,
        *,
        top_k: int | None = None,
        explain: bool = True,
    ) -> Recommendation:
        goal = goal.strip()
        category = self.resolve_category(category)
        risk = normalize_risk_choice(risk)
        band = RISK_BANDS[risk] if risk else None
        top_k = top_k or self.settings.top_k

        rewritten = self._rewrite_cached(goal)
        extra = [rewritten] if rewritten else []
        notice = None

        candidates = self.retriever.invoke(
            goal, metadata_filter=make_filter(category, band), extra_queries=extra, k=self.settings.candidate_k
        )
        if not candidates and category and band:
            # e.g. "Small Cap" + "Low": no such fund exists. Keep the category, rank by
            # closeness to the requested risk, and say so instead of silently ignoring it.
            candidates = self.retriever.invoke(
                goal, metadata_filter=make_filter(category, None), extra_queries=extra, k=self.settings.candidate_k
            )
            steps = sorted({int(d.metadata["risk_step"]) for d in candidates})
            available = [RISK_LABELS[s] for s in steps]
            notice = (
                f"No {category} are rated {risk} risk (available: {', '.join(available)}). "
                f"Showing the closest-risk funds in this category instead."
            )

        ranked = rank_funds(
            candidates, self.funds_by_code, goal, weights=self.weights, top_k=top_k, risk_band=band
        )
        horizon = parse_horizon_years(goal)
        fund_rows = [dict(r.fund) for r in ranked]

        live = (
            self.nav_client.latest_many([f["scheme_code"] for f in fund_rows]) if self.nav_client and fund_rows else {}
        )
        for row in fund_rows:
            quote = live.get(row["scheme_code"])
            if quote:
                age = nav_age_days(quote.date)
                row["nav_source"] = "stale" if age is not None and age > self.settings.nav_stale_days else "live"
                row["nav"] = quote.nav
            else:
                row["nav_source"] = "dataset" if clean_number(row.get("nav")) else "unavailable"
            row["nav_date"] = quote.date if quote else None

        explanations = (
            self._explain(fund_rows, goal, horizon, risk)
            if explain
            else [{"text": "", "source": "none"}] * len(fund_rows)
        )

        funds_out = []
        for item, row, expl in zip(ranked, fund_rows, explanations, strict=True):
            funds_out.append(
                {
                    "scheme_code": int(row["scheme_code"]),
                    "scheme_name": row["scheme_name"],
                    "category": row["category"],
                    "NAV": clean_number(row.get("nav")),
                    "nav_date": row["nav_date"],
                    "nav_source": row["nav_source"],
                    "returns_1yr": clean_number(row.get("returns_1yr")),
                    "returns_3yr": clean_number(row.get("returns_3yr")),
                    "returns_5yr": clean_number(row.get("returns_5yr")),
                    "expense_ratio": clean_number(row.get("expense_ratio")),
                    "risk_level": row["risk_label"],
                    "risk_score": clean_number(row.get("risk_level")),
                    "score": item.score,
                    "score_breakdown": item.breakdown,
                    "explanation": expl["text"],
                    "explanation_source": expl["source"],
                }
            )

        return Recommendation(
            funds=funds_out,
            query={"original": goal, "rewritten": rewritten or None, "horizon_years": horizon},
            notice=notice,
            filters={"category": category, "risk": risk, "risk_band": list(band) if band else None},
        )
