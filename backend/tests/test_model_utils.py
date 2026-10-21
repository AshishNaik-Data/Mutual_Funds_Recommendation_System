import json
import os

import pytest

from mf_recommender.config import Settings, default_adapter_dir, find_adapter
from mf_recommender.model_utils import (
    DEFAULT_BASE_MODEL,
    adapter_fits,
    has_chat_template,
    lora_target_modules,
    model_slug,
    to_chat_example,
    to_chat_prompt,
)

PHI3 = "microsoft/Phi-3-mini-4k-instruct"


class FakeChatTokenizer:
    """Mimics Phi-3-mini's chat template."""

    chat_template = "phi-3"
    eos_token = "<|endoftext|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in messages)
        return out + ("<|assistant|>\n" if add_generation_prompt else self.eos_token)


class FakeBaseTokenizer:
    chat_template = None


def test_default_model_is_phi3_mini(monkeypatch):
    monkeypatch.delenv("MF_BASE_MODEL", raising=False)
    assert DEFAULT_BASE_MODEL == PHI3
    assert Settings().base_model == PHI3
    assert Settings.from_env().base_model == PHI3


def test_chat_prompt_wraps_only_chat_models():
    assert to_chat_prompt(FakeChatTokenizer(), "Explain") == "<|user|>\nExplain<|end|>\n<|assistant|>\n"
    assert to_chat_prompt(FakeBaseTokenizer(), "Explain") == "Explain"
    assert has_chat_template(FakeChatTokenizer()) and not has_chat_template(FakeBaseTokenizer())


def test_chat_example_prompt_is_a_prefix_of_the_full_text():
    prompt, full = to_chat_example(FakeChatTokenizer(), "Explain", "It fits.")
    assert full.startswith(prompt)
    assert full[len(prompt):] == "It fits.<|end|>\n<|endoftext|>"


@pytest.mark.parametrize(
    "model_type, expected",
    [("phi3", ["qkv_proj", "o_proj"]), ("phi", ["q_proj", "k_proj", "v_proj", "dense"]), ("something-new", "all-linear")],
)
def test_lora_targets_follow_the_architecture(model_type, expected):
    assert lora_target_modules(model_type) == expected


def _adapter(folder, base, mtime):
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    os.utime(folder / "adapter_config.json", (mtime, mtime))


def test_adapter_must_match_the_base_model(tmp_path):
    _adapter(tmp_path / "lora-phi-2", "microsoft/phi-2", 1000)
    assert adapter_fits(tmp_path / "lora-phi-2", "microsoft/phi-2")
    assert not adapter_fits(tmp_path / "lora-phi-2", PHI3)
    assert adapter_fits(tmp_path / "lora-phi-2", "/some/local/copy/phi-2/")  # local path to same model


def test_find_adapter_picks_newest_matching_and_skips_checkpoints(tmp_path):
    assert find_adapter(PHI3, (tmp_path,)) is None
    _adapter(tmp_path / "lora-phi-3-old", PHI3, 1000)
    _adapter(tmp_path / "lora-phi-3-new", PHI3, 2000)
    _adapter(tmp_path / "lora-phi-2", "microsoft/phi-2", 3000)  # newest, but another model
    _adapter(tmp_path / "checkpoint-100", PHI3, 4000)  # training checkpoint
    assert find_adapter(PHI3, (tmp_path,)) == str(tmp_path / "lora-phi-3-new")
    assert find_adapter("microsoft/phi-2", (tmp_path,)) == str(tmp_path / "lora-phi-2")


def test_default_adapter_dir_names_the_model():
    assert model_slug(PHI3) == "phi-3-mini-4k-instruct"
    assert default_adapter_dir(PHI3).name == "lora-phi-3-mini-4k-instruct"


def test_explicit_adapter_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("MF_ADAPTER_PATH", str(tmp_path))
    assert Settings.from_env().adapter_path == str(tmp_path)


def test_describe_model_reports_whether_the_adapter_is_used(tmp_path):
    import dataclasses

    from mf_recommender.llm import describe_model

    adapter = tmp_path / "lora-phi-3-mini-4k-instruct"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "microsoft/Phi-3-mini-4k-instruct"})
    )
    settings = Settings(adapter_path=str(adapter))
    assert describe_model(settings, True) == {
        "llm_backend": "hf",
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "adapter": "lora-phi-3-mini-4k-instruct",
        "fine_tuned": True,
    }
    # No adapter, or one trained on another model: the untuned base model is serving.
    assert describe_model(dataclasses.replace(settings, adapter_path=None), True)["fine_tuned"] is False
    assert describe_model(dataclasses.replace(settings, base_model="microsoft/phi-2"), True)["adapter"] is None
    assert describe_model(settings, False)["llm_backend"] == "none"
