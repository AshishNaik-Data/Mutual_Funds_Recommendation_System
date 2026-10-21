"""Model-specific helpers shared by serving (llm.py) and training (scripts/train_lora.py).

The project defaults to Phi-3-mini, an instruction-tuned chat model, but nothing here is
tied to it: the chat template comes from the tokenizer, the LoRA layer names come from
the model type, and an adapter is only used with the base model it was trained on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Attention projections per architecture (config.model_type). LoRA on attention is enough
# to teach this task's format; pass --target-modules all-linear to train every layer.
LORA_TARGETS: dict[str, list[str]] = {
    "phi3": ["qkv_proj", "o_proj"],  # Phi-3 / Phi-3.5: fused q, k, v
    "phi": ["q_proj", "k_proj", "v_proj", "dense"],  # Phi-2
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gpt2": ["c_attn"],
}


def lora_target_modules(model_type: str) -> list[str] | str:
    """LoRA target layers for a model type; "all-linear" when the type is unknown."""
    return LORA_TARGETS.get(model_type, "all-linear")


def has_chat_template(tokenizer) -> bool:
    """Instruction-tuned models (Phi-3-mini) ship a chat template; base models (Phi-2) don't."""
    return bool(getattr(tokenizer, "chat_template", None))


def to_chat_prompt(tokenizer, text: str) -> str:
    """Wrap a plain prompt as one user turn, ready for the assistant's reply.

    Phi-3-mini: '<|user|>\\n{text}<|end|>\\n<|assistant|>\\n'. Base models get the text unchanged.
    """
    if not has_chat_template(tokenizer):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def to_chat_example(tokenizer, prompt: str, answer: str) -> tuple[str, str]:
    """(prompt part, full text) of one training example in the model's chat format."""
    prompt_text = to_chat_prompt(tokenizer, prompt)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], tokenize=False
    )
    return prompt_text, full_text


def model_name(path_or_id: str) -> str:
    """'microsoft/Phi-3-mini-4k-instruct' or '/x/y/Phi-3-mini-4k-instruct/' -> 'phi-3-mini-4k-instruct'."""
    return str(path_or_id).rstrip("/").split("/")[-1].lower()


def model_slug(path_or_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name(path_or_id)).strip("-")


def adapter_base_model(adapter_dir: str | Path) -> str | None:
    """The base model an adapter was trained on, from its adapter_config.json."""
    try:
        cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    except (OSError, ValueError):
        return None
    return cfg.get("base_model_name_or_path") or None


def adapter_fits(adapter_dir: str | Path, base_model: str) -> bool:
    """True if the adapter was trained on ``base_model``: a LoRA adapter only fits its own base."""
    base = adapter_base_model(adapter_dir)
    return base is not None and model_name(base) == model_name(base_model)
