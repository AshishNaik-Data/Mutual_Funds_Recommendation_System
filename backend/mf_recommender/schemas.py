"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RecommendRequest(BaseModel):
    goal: str = Field(..., min_length=5, max_length=500, description="Investment goal in plain English")
    category: str | None = Field(None, description='Sub-category, or null / "" / "Any" to search all')
    risk: Literal["Low", "Medium", "High"] | None = Field(None, description="Risk preference, or null for any")
    top_k: int = Field(5, ge=1, le=10)
    explain: bool = Field(True, description="Generate explanations (slower with a local LLM)")

    @field_validator("goal")
    @classmethod
    def _strip_goal(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("goal must be at least 5 characters")
        return v

    @field_validator("risk", mode="before")
    @classmethod
    def _normalise_risk(cls, v: object) -> object:
        if v is None or (isinstance(v, str) and v.strip().lower() in {"", "any"}):
            return None
        return v.strip().capitalize() if isinstance(v, str) else v


class ScoreBreakdown(BaseModel):
    relevance: float
    returns: float
    cost: float
    horizon_fit: float
    risk_fit: float


class FundRecommendation(BaseModel):
    scheme_code: int
    scheme_name: str
    category: str
    NAV: float | None
    nav_date: str | None
    nav_source: Literal["live", "dataset", "unavailable"]
    returns_1yr: float | None
    returns_3yr: float | None
    returns_5yr: float | None
    expense_ratio: float | None
    risk_level: str = Field(..., description="Riskometer label, e.g. 'Moderately High'")
    risk_score: float | None = Field(None, description="Raw dataset value, 0.0 (Low) to 0.5 (Very High)")
    score: float
    score_breakdown: ScoreBreakdown
    explanation: str
    explanation_source: Literal["llm", "template", "none"]


class QueryInfo(BaseModel):
    original: str
    rewritten: str | None
    horizon_years: float | None


class RecommendResponse(BaseModel):
    funds: list[FundRecommendation]
    query: QueryInfo
    notice: str | None = None
    filters: dict


class CategoriesResponse(BaseModel):
    categories: list[str]
    risk_levels: list[str]
