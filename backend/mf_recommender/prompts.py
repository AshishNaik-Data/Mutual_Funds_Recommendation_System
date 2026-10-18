"""Prompt templates.

EXPLAIN_TEMPLATE is shared with scripts/train_lora.py so the fine-tuned model sees at
inference exactly the format it was trained on (the notebook trained with
"...\\n\\nAnswer: <response>" but prompted without "Answer:" at inference).
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

REWRITE_TEMPLATE = (
    "You are a financial assistant. Rewrite the investor's goal as one short search query "
    "describing the kind of mutual fund they need: asset class, time horizon, risk and purpose. "
    "Do not name specific funds.\n"
    "Investor goal: {goal}\n"
    "Search query:"
)

EXPLAIN_TEMPLATE = (
    "User goal: {goal}\n"
    "I'm looking for a mutual fund in the '{category}' category with '{risk_label}' risk. "
    "Explain if the following fund is a suitable option, using only the facts below:\n\n"
    "Scheme Name: {scheme_name}\n"
    "NAV: {nav_text}\n"
    "1Y Return: {returns_1yr}%\n"
    "3Y Return: {returns_3yr}%\n"
    "5Y Return: {returns_5yr}%\n"
    "Expense Ratio: {expense_ratio}%\n"
    "Risk Level: {risk_label}\n"
    "Peer comparison: {peer_summary}\n"
    "Investor horizon: {horizon}; category suits: {suggested_horizon}\n\n"
    "Answer:"
)

REWRITE_PROMPT = PromptTemplate.from_template(REWRITE_TEMPLATE)
EXPLAIN_PROMPT = PromptTemplate.from_template(EXPLAIN_TEMPLATE)
