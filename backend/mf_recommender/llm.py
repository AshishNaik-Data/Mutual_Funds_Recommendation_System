"""LLM factory: a local Hugging Face model (+ LoRA adapter), any LangChain chat model, or none.

Default: microsoft/Phi-3-mini-4k-instruct, an instruction-tuned chat model, so prompts are
wrapped in its chat template (the same wrapping the training script uses).
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.runnables import Runnable, RunnableLambda

from .config import Settings
from .model_utils import adapter_base_model, adapter_fits, has_chat_template, to_chat_prompt

log = logging.getLogger(__name__)


def build_llm(settings: Settings) -> Runnable | None:
    backend = settings.llm_backend.lower()
    if backend == "none":
        return None
    if backend == "hf":
        return build_hf_llm(
            settings.base_model,
            adapter_path=settings.adapter_path,
            max_new_tokens=settings.max_new_tokens,
            batch_size=settings.llm_batch_size,
        )
    if backend == "chat":
        if not settings.chat_model:
            raise ValueError('MF_LLM_BACKEND=chat needs MF_CHAT_MODEL, e.g. "openai:<model-name>"')
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The chat backend needs `pip install langchain` plus the provider package "
                "(e.g. langchain-openai or langchain-anthropic)."
            ) from exc
        return init_chat_model(settings.chat_model, temperature=0)
    raise ValueError(f"Unknown MF_LLM_BACKEND {settings.llm_backend!r} (use hf, chat or none)")


def usable_adapter(adapter_path: str | None, base_model: str) -> str | None:
    """The adapter that will actually be loaded: None if unset or trained on another model."""
    return adapter_path if adapter_path and adapter_fits(adapter_path, base_model) else None


def describe_model(settings: Settings, llm_loaded: bool) -> dict:
    """What writes the explanations, for /health, the UI footer and the notebook.

    "fine_tuned": False with backend "hf" means the base model is running without a LoRA
    adapter, so explanations come from the untuned model (and mostly fail the quality gate).
    """
    info = {"llm_backend": "none", "base_model": None, "adapter": None, "fine_tuned": False}
    if not llm_loaded:
        return info
    if settings.llm_backend == "hf":
        adapter = usable_adapter(settings.adapter_path, settings.base_model)
        info.update(base_model=settings.base_model, adapter=Path(adapter).name if adapter else None,
                    fine_tuned=bool(adapter), llm_backend="hf")
    elif settings.llm_backend == "chat":
        info.update(llm_backend="chat", base_model=settings.chat_model)
    else:
        info.update(llm_backend="custom")  # an LLM object passed in directly (tests, experiments)
    return info


def _prompt_text(prompt) -> str:
    """A LangChain prompt value (or plain string) as text."""
    return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)


def build_hf_llm(
    base_model: str,
    *,
    adapter_path: str | None = None,
    max_new_tokens: int = 120,
    batch_size: int = 4,
) -> Runnable:
    import torch
    import transformers
    from langchain_huggingface import HuggingFacePipeline
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if adapter_path and not usable_adapter(adapter_path, base_model):
        # A LoRA adapter only fits the model it was trained on; loading it onto another
        # model fails. Run the base model instead and say why.
        log.warning(
            "Ignoring adapter %s: it was trained on %s, not %s. Retrain with scripts/train_lora.py.",
            adapter_path, adapter_base_model(adapter_path), base_model,
        )
        adapter_path = None

    cuda = torch.cuda.is_available()
    # T4 GPUs have no bfloat16, so float16 there; float32 on CPU.
    dtype = torch.float16 if cuda else torch.float32
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation with decoder-only models

    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto" if cuda else None, **{dtype_kw: dtype}
    )
    if not adapter_path:
        log.warning("No LoRA adapter for %s: explanations come from the untuned model. "
                    "Train one with scripts/train_lora.py.", base_model)
    else:
        from peft import PeftModel

        log.info("Loading LoRA adapter from %s", adapter_path)
        # Merging the adapter makes generation as fast as the base model.
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,  # only the new text, no prompt echo
        pad_token_id=tokenizer.pad_token_id,
    )
    llm = HuggingFacePipeline(pipeline=generator, batch_size=batch_size)

    if has_chat_template(tokenizer):
        # Chat models (Phi-3-mini) answer best when the prompt is a user turn in their own
        # format: <|user|>\n...<|end|>\n<|assistant|>\n. Generation stops at <|end|>.
        log.info("Using %s's chat template", base_model)
        return RunnableLambda(lambda prompt: to_chat_prompt(tokenizer, _prompt_text(prompt))) | llm
    return llm
