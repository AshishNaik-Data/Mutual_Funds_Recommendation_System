"""LCEL chains for query rewriting and fund explanations."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from .explanations import describe_fund, is_grounded, prompt_inputs
from .prompts import EXPLAIN_PROMPT, REWRITE_PROMPT

# Small base models keep writing past the answer (a new "User goal:", another fund, ...).
_STOP_MARKERS = (
    "\n\n", "\nUser goal", "User goal:", "\nInvestor goal", "\nScheme Name", "\nQuestion",
    "\nInstruction", "\nAnswer:", "<|endoftext|>", "\n###",
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


def is_usable_explanation(text: str, facts: dict[str, str]) -> bool:
    """Quality gate for model-written explanations: real sentences, not repetitive,
    and every percentage it cites is present in the facts it was given."""
    words = text.split()
    if len(text) < 40 or len(words) < 6:
        return False
    if sum(len(w) for w in words) / len(words) > 12:  # run-together junk tokens
        return False
    return not is_degenerate(text) and is_grounded(text, facts)


def clean_rewrite(text: str) -> str:
    line = _cut(text).splitlines()[0] if _cut(text) else ""
    line = re.sub(r"^(search query|query|rewritten query)\s*:\s*", "", line.strip(), flags=re.I)
    line = line.strip().strip("\"'`").strip()[:200]
    real_words = re.findall(r"[A-Za-z]{2,}", line)
    return line if len(real_words) >= 2 and not is_degenerate(line) else ""


def build_rewrite_chain(llm: BaseLanguageModel | None) -> Runnable | None:
    """goal (str) -> search query (str); returns "" when the model gives nothing usable."""
    if llm is None:
        return None
    return {"goal": RunnablePassthrough()} | REWRITE_PROMPT | llm | StrOutputParser() | RunnableLambda(clean_rewrite)


def _template(x: dict[str, Any]) -> dict[str, str]:
    return {"text": describe_fund(x["fund"], x["goal"], x["horizon_years"]), "source": "template"}


def _inputs(x: dict[str, Any]) -> dict[str, str]:
    return prompt_inputs(x["fund"], x["goal"], x["horizon_years"])


def _choose(x: dict[str, Any]) -> dict[str, str]:
    text = x["text"]
    if is_usable_explanation(text, x["inputs"]):
        return {"text": text, "source": "llm"}
    return x["fallback"]


def build_explain_chain(llm: BaseLanguageModel | None) -> Runnable:
    """{"fund": dict, "goal": str, "horizon_years": float | None} -> {"text", "source"}.

    With an LLM: prompt -> model -> clean-up, then a quality gate. If the answer is
    empty, too short, repetitive, cites a percentage that isn't in the fund facts, or the
    model raises, the deterministic explanation is used instead.
    """
    template = RunnableLambda(_template)
    if llm is None:
        return template
    generate = RunnableLambda(_inputs) | EXPLAIN_PROMPT | llm | StrOutputParser() | RunnableLambda(clean_generation)
    chain = RunnableParallel(text=generate, inputs=RunnableLambda(_inputs), fallback=template) | RunnableLambda(_choose)
    return chain.with_fallbacks([template])
