"""Plain-language profiles of the SEBI mutual fund sub-categories in the dataset.

The original index embedded only "<scheme name> - <category>", which gives a semantic
search engine almost nothing to match an investor's goal against. These profiles add
what each category is for, its typical holding period and any lock-in, so a query such
as "park my emergency money for a few months" can land on liquid / overnight funds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryProfile:
    asset_class: str
    summary: str
    horizon: str  # human-readable suggested holding period
    min_years: float  # shortest sensible holding period
    max_years: float | None = None  # beyond this the category is a poor fit (short-term debt)
    lock_in: str | None = None


CATEGORY_PROFILES: dict[str, CategoryProfile] = {
    # --- Debt ---------------------------------------------------------------
    "Overnight Mutual Funds": CategoryProfile(
        "debt", "Invests in overnight (one-day) securities. Lowest-risk fund type, used to park "
        "surplus cash for a few days with near-zero volatility.", "1 day to a few weeks", 0.0, 0.25),
    "Liquid Mutual Funds": CategoryProfile(
        "debt", "Invests in money-market instruments maturing within 91 days. Suited to an "
        "emergency fund or parking cash safely with quick withdrawal and stable returns.",
        "a few days to 3 months", 0.0, 0.5),
    "Money Market Funds": CategoryProfile(
        "debt", "Invests in money-market instruments maturing within one year. Low risk, "
        "short-term parking of cash with slightly higher returns than liquid funds.",
        "3 to 12 months", 0.25, 1.5),
    "Ultra Short Duration Funds": CategoryProfile(
        "debt", "Debt fund with a portfolio duration of 3 to 6 months. Low risk, stable "
        "short-term income.", "3 months to 1 year", 0.25, 1.5),
    "Low Duration Funds": CategoryProfile(
        "debt", "Debt fund with a portfolio duration of 6 to 12 months. Low interest-rate risk, "
        "steady short-term income.", "6 to 12 months", 0.5, 2.0),
    "Short Duration Funds": CategoryProfile(
        "debt", "Debt fund with a portfolio duration of 1 to 3 years. Stable income with modest "
        "interest-rate risk.", "1 to 3 years", 1.0, 4.0),
    "Medium Duration Funds": CategoryProfile(
        "debt", "Debt fund with a portfolio duration of 3 to 4 years. Regular income, moderate "
        "interest-rate risk.", "3 to 4 years", 3.0),
    "Medium to Long Duration Funds": CategoryProfile(
        "debt", "Debt fund with a portfolio duration of 4 to 7 years. Income-oriented, sensitive "
        "to interest-rate changes.", "4 years or more", 4.0),
    "Dynamic Bond": CategoryProfile(
        "debt", "Debt fund whose manager moves between short and long bonds based on the "
        "interest-rate outlook. Income with some interest-rate risk.", "3 years or more", 3.0),
    "Corporate Bond Mutual Funds": CategoryProfile(
        "debt", "Invests mostly in highest-rated (AA+ and above) corporate bonds. Steady income "
        "with high credit quality.", "1 to 3 years", 1.0),
    "Credit Risk Funds": CategoryProfile(
        "debt", "Invests mostly in lower-rated corporate bonds for higher yield. Higher income "
        "potential with real default risk.", "3 years or more", 3.0),
    "Banking and PSU Mutual Funds": CategoryProfile(
        "debt", "Invests mostly in bonds of banks, public sector undertakings and public "
        "financial institutions. High credit quality, stable income.", "1 to 3 years", 1.0),
    "Gilt Mutual Funds": CategoryProfile(
        "debt", "Invests mostly in government securities (gilts). No credit risk, but prices move "
        "with interest rates.", "3 years or more", 3.0),
    "Floater Mutual Funds": CategoryProfile(
        "debt", "Invests mostly in floating-rate bonds whose coupons reset with interest rates. "
        "Low interest-rate risk, benefits when rates rise.", "1 to 3 years", 1.0),
    # --- Hybrid -------------------------------------------------------------
    "Arbitrage Mutual Funds": CategoryProfile(
        "hybrid", "Earns low-risk returns from price differences between the cash and futures "
        "markets. Debt-like stability with equity taxation; short-term parking of money.",
        "3 to 12 months", 0.25, 2.0),
    "Equity Savings Mutual Funds": CategoryProfile(
        "hybrid", "Mix of equity, arbitrage and debt. Moderate income with a little growth and "
        "lower volatility than pure equity.", "2 to 3 years or more", 2.0),
    "Conservative Hybrid Mutual Funds": CategoryProfile(
        "hybrid", "Mostly debt (75-90%) with a small equity portion (10-25%). Regular income with "
        "a little growth.", "2 to 3 years or more", 2.0),
    "Aggressive Hybrid Mutual Funds": CategoryProfile(
        "hybrid", "Mostly equity (65-80%) with the rest in debt. Long-term growth with a cushion "
        "against market falls.", "3 to 5 years or more", 3.0),
    "Dynamic Asset Allocation or Balanced Advantage": CategoryProfile(
        "hybrid", "Shifts between equity and debt based on market valuations. Smoother, balanced "
        "growth with moderate risk.", "3 to 5 years or more", 3.0),
    "Multi Asset Allocation Mutual Funds": CategoryProfile(
        "hybrid", "Invests in at least three asset classes such as equity, debt and gold. "
        "Diversified growth with lower dependence on any one market.", "3 to 5 years or more", 3.0),
    # --- Equity -------------------------------------------------------------
    "Large Cap Mutual Funds": CategoryProfile(
        "equity", "Invests mostly in India's 100 largest companies (blue chips). Long-term capital "
        "growth; the most stable equity category.", "5 years or more", 5.0),
    "Large & Mid Cap Funds": CategoryProfile(
        "equity", "Invests in both large and mid-sized companies. Long-term growth with higher "
        "volatility than large caps.", "5 years or more", 5.0),
    "Mid Cap Mutual Funds": CategoryProfile(
        "equity", "Invests mostly in mid-sized companies (ranked 101-250). High long-term growth "
        "potential with high volatility.", "5 to 7 years or more", 5.0),
    "Small Cap Mutual Funds": CategoryProfile(
        "equity", "Invests mostly in small companies (ranked 251 and below). Highest growth "
        "potential and the sharpest falls; aggressive investors only.", "7 years or more", 7.0),
    "Multi Cap Funds": CategoryProfile(
        "equity", "Invests at least 25% each in large, mid and small caps. Diversified long-term "
        "equity growth.", "5 years or more", 5.0),
    "Flexi Cap Funds": CategoryProfile(
        "equity", "Equity fund free to invest across large, mid and small caps. Diversified "
        "long-term wealth creation.", "5 years or more", 5.0),
    "Focused Funds": CategoryProfile(
        "equity", "Concentrated equity portfolio of at most 30 stocks. Long-term growth with "
        "higher stock-specific risk.", "5 years or more", 5.0),
    "Value Funds": CategoryProfile(
        "equity", "Follows a value-investing style, buying undervalued companies. Long-term growth "
        "for patient investors.", "5 years or more", 5.0),
    "Contra Funds": CategoryProfile(
        "equity", "Follows a contrarian style, buying out-of-favour stocks. Long-term growth for "
        "patient investors.", "5 years or more", 5.0),
    "Dividend Yield Funds": CategoryProfile(
        "equity", "Invests mostly in high dividend-yield companies. Long-term growth from "
        "relatively defensive, cash-generating businesses.", "5 years or more", 5.0),
    "Sectoral / Thematic Mutual Funds": CategoryProfile(
        # No example themes here: listing "banking, pharma, ..." would make every thematic
        # fund match every theme. The scheme name carries the actual sector.
        "equity", "Concentrated in a single sector or investment theme. Very high risk; for "
        "investors with a specific view on that sector.", "5 years or more", 5.0),
    "ELSS Mutual Funds": CategoryProfile(
        "equity", "Equity Linked Savings Scheme: tax-saving equity fund (the 80C deduction, "
        "available under the old tax regime). Long-term growth with a mandatory lock-in.",
        "at least 3 years (lock-in), ideally 5 or more", 3.0, lock_in="3-year lock-in"),
    "Index Funds": CategoryProfile(
        "equity", "Passive fund that tracks a market index such as the Nifty 50 or Sensex. Low "
        "cost, market-matching long-term returns.", "5 years or more", 5.0),
    # --- Fund of funds and solution-oriented --------------------------------
    "FoFs Domestic": CategoryProfile(
        "fund of funds", "Invests in other domestic mutual funds for one-stop diversification. "
        "Risk depends on the underlying funds.", "3 years or more", 3.0),
    "Retirement Funds": CategoryProfile(
        "solution-oriented", "Built for retirement planning and building a pension corpus. "
        "Long-term growth with a lock-in until retirement age or five years.",
        "5 years or more, until retirement", 5.0,
        lock_in="5-year lock-in (or until retirement age)"),
    "Childrens Funds": CategoryProfile(
        "solution-oriented", "Built for a child's future goals such as higher education or "
        "marriage. Long-term growth with a lock-in until the child turns 18 or five years.",
        "5 years or more, until the child turns 18", 5.0,
        lock_in="5-year lock-in (or until the child turns 18)"),
}

GENERIC_PROFILE = CategoryProfile(
    "mutual fund", "Mutual fund scheme.", "3 years or more", 3.0)


def profile_for(category: str) -> CategoryProfile:
    return CATEGORY_PROFILES.get(category, GENERIC_PROFILE)


_SHORT_NAMES = {
    "FoFs Domestic": "domestic fund-of-funds",
    "Childrens Funds": "children's",
    "Dynamic Asset Allocation or Balanced Advantage": "balanced advantage",
    "Sectoral / Thematic Mutual Funds": "sectoral/thematic",
}


def short_name(category: str) -> str:
    """'Large Cap Mutual Funds' -> 'Large Cap', for phrases like 'a Large Cap fund'."""
    if category in _SHORT_NAMES:
        return _SHORT_NAMES[category]
    for suffix in (" Mutual Funds", " Funds", " Fund"):
        if category.endswith(suffix):
            return category[: -len(suffix)]
    return category


def with_article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase
