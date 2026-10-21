"""LCEL chains for query rewriting and fund explanations."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from .explanations import cites_a_percentage, contradictions, describe_fund, is_grounded, prompt_inputs
from .prompts import EXPLAIN_PROMPT, REWRITE_PROMPT

log = logging.getLogger(__name__)

# Small models can keep writing past the answer (a new "User goal:", another fund, a new
# chat turn ...); cut there. The <|...|> markers are Phi-3's chat tokens, in case they leak.
_STOP_MARKERS = (
    "\n\n", "\nUser goal", "User goal:", "\nInvestor goal", "\nScheme Name", "\nQuestion",
    "\nInstruction", "\nAnswer:", "<|endoftext|>", "<|end|>", "<|user|>", "<|assistant|>", "\n###",
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _cut(text: str) -> str:
    text = text.strip()
    cut = len(text)
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def clean_generation(text: str, max_sentences: int = 4) -> str:
    text = re.sub(r"\s+", " ", _cut(text)).strip()
    sentences = _SENTENCE_END.split(text)
    text = " ".join(sentences[:max_sentences]).strip()
    # Drop a trailing half-sentence left by the token limit.
    if len(sentences) > 1 and text and text[-1] not in ".!?":
        text = text[: text.rfind(".") + 1] if "." in text else text
    return text


def is_degenerate(text: str) -> bool:
    """Repetitive output ("of of of ...", the same clause looping) from a weak or broken model."""
    words = re.findall(r"\w+", text.lower())
    return len(words) >= 8 and len(set(words)) / len(words) < 0.35


def explanation_problem(text: str, facts: dict[str, str], fund: dict | None = None) -> str | None:
    """Why a model-written explanation can't be shown, or None if it passes.

    It must be real sentences, not repetitive, cite at least one figure, cite only
    percentages present in the facts it was given, and (when ``fund`` is passed) make no
    claim about cost or returns that the peer comparison contradicts.
    """
    words = text.split()
    if len(text) < 40 or len(words) < 6:
        return "too short"
    if sum(len(w) for w in words) / len(words) > 12:
        return "run-together junk tokens"
    if is_degenerate(text):
        return "repetitive"
    if not is_grounded(text, facts):
        return "cites a percentage that is not in the fund facts"
    if not cites_a_percentage(text):
        return "cites none of the fund's figures"
    if fund is not None and (found := contradictions(text, fund)):
        return found[0]
    return None


def is_usable_explanation(text: str, facts: dict[str, str], fund: dict | None = None) -> bool:
    return explanation_problem(text, facts, fund) is None


def clean_rewrite(text: str) -> str:
    line = _cut(text).splitlines()[0] if _cut(text) else ""
    line = re.sub(r"^(search query|query|rewritten query)\s*:\s*", "", line.strip(), flags=re.I)
    line = line.strip().strip("\"'`").strip()[:200]
    real_words = re.findall(r"[A-Za-z]{2,}", line)
    return line if len(real_words) >= 2 and not is_degenerate(line) else ""


def build_rewrite_chain(llm: Runnable | None) -> Runnable | None:
    """goal (str) -> search query (str); returns "" when the model gives nothing usable."""
    if llm is None:
        return None
    return {"goal": RunnablePassthrough()} | REWRITE_PROMPT | llm | StrOutputParser() | RunnableLambda(clean_rewrite)


def _template(x: dict[str, Any]) -> dict[str, str]:
    text = describe_fund(x["fund"], x["goal"], x["horizon_years"], x.get("risk"))
    return {"text": text, "source": "template"}


def _inputs(x: dict[str, Any]) -> dict[str, str]:
    return prompt_inputs(x["fund"], x["goal"], x["horizon_years"], x.get("risk"))


def _choose(x: dict[str, Any]) -> dict[str, str]:
    problem = explanation_problem(x["text"], x["inputs"], x["fund"])
    if problem is None:
        return {"text": x["text"], "source": "llm"}
    log.info("Model explanation for %s rejected (%s): %r", x["fund"]["scheme_name"], problem, x["text"][:200])
    return x["fallback"]


def build_explain_chain(llm: Runnable | None) -> Runnable:
    """{"fund": dict, "goal": str, "horizon_years": float | None, "risk": str | None} -> {"text", "source"}.

    With an LLM: prompt -> model -> clean-up, then a quality gate (explanation_problem). If
    the answer is junk, cites no figures or invented ones, contradicts the peer comparison,
    or the model raises, the deterministic explanation is used instead.
    """
    template = RunnableLambda(_template)
    if llm is None:
        return template
    generate = RunnableLambda(_inputs) | EXPLAIN_PROMPT | llm | StrOutputParser() | RunnableLambda(clean_generation)
    chain = RunnableParallel(
        text=generate, inputs=RunnableLambda(_inputs), fund=RunnableLambda(lambda x: x["fund"]), fallback=template
    ) | RunnableLambda(_choose)
    return chain.with_fallbacks([template])
