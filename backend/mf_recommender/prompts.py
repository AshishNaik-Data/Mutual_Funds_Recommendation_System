"""Prompt templates.

EXPLAIN_TEMPLATE is shared with scripts/train_lora.py so the fine-tuned model sees at
inference exactly the format it was trained on (the notebook trained with
"...\\n\\nAnswer: <response>" but prompted without "Answer:" at inference).
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

# Examples show the model to match the asset class to the horizon: without them, Phi-3
# rewrote "I want a short term investment" as "Short-term growth equity mutual funds".
REWRITE_TEMPLATE = (
    "You are a financial assistant. Rewrite the investor's goal as one short search query "
    "describing the kind of mutual fund they need: asset class, time horizon, risk and purpose. "
    "Match the asset class to the horizon: under 3 years means debt, liquid or money market "
    "funds, not equity. Do not name specific funds. Reply with the query only.\n\n"
    "Investor goal: Park my bonus safely for 6 months\n"
    "Search query: liquid or ultra short duration debt fund, low risk, 6 months\n\n"
    "Investor goal: Build a retirement corpus over 20 years\n"
    "Search query: diversified equity fund for long-term wealth creation, 20 years\n\n"
    "Investor goal: Save tax and grow my money\n"
    "Search query: ELSS tax saving equity fund with 3-year lock-in\n\n"
    "Investor goal: {goal}\n"
    "Search query:"
)

# The investor's own risk choice and how the fund compares with it are given as facts. (The
# previous version told the model the investor wanted the fund's risk level, so with "High"
# selected and only Moderate funds available it wrote "you are seeking moderate risk".)
EXPLAIN_TEMPLATE = (
    "User goal: {goal}\n"
    "Risk I asked for: {requested_risk}\n"
    "Explain if the following '{category}' fund is a suitable option, using only the facts below:\n\n"
    "Scheme Name: {scheme_name}\n"
    "NAV: {nav_text}\n"
    "1Y Return: {returns_1yr}%\n"
    "3Y Return: {returns_3yr}%\n"
    "5Y Return: {returns_5yr}%\n"
    "Expense Ratio: {expense_ratio}%\n"
    "Risk Level: {risk_label} ({risk_fit})\n"
    "Peer comparison: {peer_summary}\n"
    "Investor horizon: {horizon}; category suits: {suggested_horizon}\n\n"
    "Answer:"
)

REWRITE_PROMPT = PromptTemplate.from_template(REWRITE_TEMPLATE)
EXPLAIN_PROMPT = PromptTemplate.from_template(EXPLAIN_TEMPLATE)
