"""Loading and cleaning the fund dataset, and turning funds into LangChain Documents."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.documents import Document

from .categories import profile_for

# The dataset stores risk as 0.0-0.5 in steps of 0.1: six levels, which matches the SEBI
# riskometer. (Overnight funds are all 0.0, equity funds are all 0.5.)
RISK_LABELS: dict[int, str] = {
    0: "Low",
    1: "Low to Moderate",
    2: "Moderate",
    3: "Moderately High",
    4: "High",
    5: "Very High",
}

# What the UI's three choices mean on that six-step scale (inclusive).
RISK_BANDS: dict[str, tuple[int, int]] = {
    "Low": (0, 1),
    "Medium": (2, 3),
    "High": (4, 5),
}

RETURN_COLUMNS = ("returns_1yr", "returns_3yr", "returns_5yr")

_COLUMN_MAP = {"Scheme Codes": "scheme_code", "Sub_Category": "category", "NAV": "nav"}
_REQUIRED = {"scheme_name", "scheme_code", "category", "expense_ratio", "risk_level", *RETURN_COLUMNS}


def normalize_risk_choice(risk: str | None) -> str | None:
    """Map user input like 'low' / 'MEDIUM' to a RISK_BANDS key (None means no filter)."""
    if risk is None or not str(risk).strip() or str(risk).strip().lower() == "any":
        return None
    for name in RISK_BANDS:
        if name.lower() == str(risk).strip().lower():
            return name
    raise ValueError(f"Unknown risk level {risk!r}; expected one of {list(RISK_BANDS)}")


def _peer_share(df: pd.DataFrame, column: str, *, higher_is_better: bool) -> pd.Series:
    """Share of *other* funds in the same category that this fund beats on ``column``.

    1.0 = best in category, 0.0 = worst, 0.5 when there are no peers or the value is missing.
    """
    values = df[column] if higher_is_better else -df[column]
    rank = values.groupby(df["category"]).rank(method="min")  # 1 = worst
    peers = df.groupby("category")[column].transform("count") - 1
    share = (rank - 1) / peers.where(peers > 0)
    return share.fillna(0.5).clip(0.0, 1.0)


def load_funds(path: str | Path) -> pd.DataFrame:
    """Read the CSV, normalise column names and types, and add risk labels and peer stats."""
    df = pd.read_csv(path).rename(columns=_COLUMN_MAP)
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["scheme_name"] = df["scheme_name"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["scheme_name"] != "") & (df["category"] != "")].copy()
    df["scheme_code"] = pd.to_numeric(df["scheme_code"], errors="coerce")
    df = df.dropna(subset=["scheme_code"]).drop_duplicates(subset="scheme_code")
    df["scheme_code"] = df["scheme_code"].astype(int)

    for col in ("expense_ratio", "risk_level", *RETURN_COLUMNS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "nav" not in df.columns:
        df["nav"] = np.nan
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df.loc[df["nav"] <= 0, "nav"] = np.nan  # failed NAV fetches were saved as 0.0

    df["risk_step"] = (df["risk_level"].fillna(0.5) * 10).round().clip(0, 5).astype(int)
    df["risk_label"] = df["risk_step"].map(RISK_LABELS)

    for col in RETURN_COLUMNS:
        df[f"{col}_beats"] = _peer_share(df, col, higher_is_better=True)
    df["expense_beats"] = _peer_share(df, "expense_ratio", higher_is_better=False)
    df["peer_count"] = df.groupby("category")["scheme_code"].transform("count") - 1

    return df.reset_index(drop=True)


def fund_text(row: dict) -> str:
    """Text that gets embedded / keyword-indexed for one fund."""
    profile = profile_for(row["category"])
    asset = profile.asset_class if profile.asset_class.endswith("funds") else f"{profile.asset_class} fund"
    parts = [
        f"{row['scheme_name']}.",
        f"Category: {row['category']} ({asset}).",
        profile.summary,
        f"Suggested holding period: {profile.horizon}.",
        f"Riskometer: {row['risk_label']} risk.",
    ]
    if profile.lock_in:
        parts.append(f"Lock-in: {profile.lock_in}.")
    return " ".join(parts)


def fund_documents(df: pd.DataFrame) -> list[Document]:
    """One Document per fund. Numbers stay in the DataFrame so NAV/returns can change
    without re-embedding; metadata holds only what retrieval filters on."""
    docs = []
    for row in df.to_dict("records"):
        docs.append(
            Document(
                id=str(row["scheme_code"]),
                page_content=fund_text(row),
                metadata={
                    "scheme_code": int(row["scheme_code"]),
                    "scheme_name": row["scheme_name"],
                    "category": row["category"],
                    "risk_step": int(row["risk_step"]),
                },
            )
        )
    return docs


def clean_number(value: object) -> float | None:
    """NaN/None -> None, otherwise a plain float (JSON-safe)."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number
