"""LLM factory: local Hugging Face model (+ LoRA adapter), any LangChain chat model, or none."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseLanguageModel

from .config import Settings

log = logging.getLogger(__name__)


def build_llm(settings: Settings) -> BaseLanguageModel | None:
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


def build_hf_llm(
    base_model: str,
    *,
    adapter_path: str | None = None,
    max_new_tokens: int = 120,
    batch_size: int = 4,
) -> BaseLanguageModel:
    import torch
    import transformers
    from langchain_huggingface import HuggingFacePipeline
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation with decoder-only models

    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto" if cuda else None, **{dtype_kw: dtype}
    )
    if adapter_path:
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
        return_full_text=False,  # no more output.replace(prompt, "")
        pad_token_id=tokenizer.pad_token_id,
    )
    return HuggingFacePipeline(pipeline=generator, batch_size=batch_size)
