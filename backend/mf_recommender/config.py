"""Runtime configuration, read from environment variables (all prefixed ``MF_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .model_utils import DEFAULT_BASE_MODEL, adapter_fits, model_slug

# Every default path is relative to the project folder, wherever it lives (Drive, laptop, ...).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "updated_data_with_nav.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_INDEX_DIR = ARTIFACTS_DIR / "faiss_index"


def default_adapter_dir(base_model: str) -> Path:
    """Where train_lora.py saves an adapter: artifacts/lora-<model>, e.g. artifacts/lora-phi-3-mini-4k-instruct."""
    return ARTIFACTS_DIR / f"lora-{model_slug(base_model)}"


def find_adapter(
    base_model: str | None = None, search_dirs: tuple[Path, ...] = (ARTIFACTS_DIR, PROJECT_ROOT)
) -> str | None:
    """Newest LoRA adapter folder (one holding adapter_config.json) directly inside
    ``search_dirs``, skipping training checkpoints. With ``base_model``, only adapters
    trained on that model count: an adapter for another model would fail to load."""
    found = []
    for folder in search_dirs:
        if folder.is_dir():
            found += [p.parent for p in folder.glob("*/adapter_config.json") if "checkpoint" not in p.parent.name]
    if base_model is not None:
        found = [p for p in found if adapter_fits(p, base_model)]
    if not found:
        return None
    return str(max(found, key=lambda p: (p / "adapter_config.json").stat().st_mtime))


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"MF_{name}")
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class Settings:
    # Data and index
    data_path: Path = DEFAULT_DATA_PATH
    index_dir: Path = DEFAULT_INDEX_DIR
    # "hashing" is an offline, lexical-only fallback (no model download); use it for tests.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM: "hf" (local Hugging Face model + optional LoRA adapter),
    #      "chat" (any LangChain chat model via init_chat_model), or "none" (templates only).
    llm_backend: str = "hf"
    base_model: str = DEFAULT_BASE_MODEL  # microsoft/Phi-3-mini-4k-instruct
    adapter_path: str | None = None
    chat_model: str | None = None  # e.g. "openai:<model-name>" when llm_backend == "chat"
    max_new_tokens: int = 120
    llm_batch_size: int = 5
    rewrite_query: bool = True

    # Live NAV (api.mfapi.in)
    live_nav: bool = True
    nav_ttl_seconds: int = 6 * 3600
    nav_timeout_seconds: float = 8.0
    # NAVs are published every business day; one older than this belongs to a plan that has
    # probably been closed or merged (the dataset has some old scheme codes), so it is
    # shown as "last published <date>" rather than "live".
    nav_stale_days: int = 10

    # Retrieval and ranking
    candidate_k: int = 30
    top_k: int = 5
    weight_relevance: float = 0.40
    weight_returns: float = 0.40
    weight_cost: float = 0.20

    # API
    cors_origins: tuple[str, ...] = field(default_factory=lambda: ("*",))

    @classmethod
    def from_env(cls) -> Settings:
        origins = _env("CORS_ORIGINS", "*") or "*"
        base_model = _env("BASE_MODEL", cls.base_model) or cls.base_model
        return cls(
            data_path=Path(_env("DATA_PATH", str(DEFAULT_DATA_PATH))),
            index_dir=Path(_env("INDEX_DIR", str(DEFAULT_INDEX_DIR))),
            embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
            llm_backend=(_env("LLM_BACKEND", cls.llm_backend) or "hf").lower(),
            base_model=base_model,
            # auto: the newest adapter in artifacts/ that was trained on this base model
            adapter_path=_env("ADAPTER_PATH") or find_adapter(base_model),
            chat_model=_env("CHAT_MODEL"),
            max_new_tokens=_env_int("MAX_NEW_TOKENS", cls.max_new_tokens),
            llm_batch_size=_env_int("LLM_BATCH_SIZE", cls.llm_batch_size),
            rewrite_query=_env_bool("REWRITE_QUERY", cls.rewrite_query),
            live_nav=_env_bool("LIVE_NAV", cls.live_nav),
            nav_ttl_seconds=_env_int("NAV_TTL_SECONDS", cls.nav_ttl_seconds),
            nav_timeout_seconds=_env_float("NAV_TIMEOUT_SECONDS", cls.nav_timeout_seconds),
            nav_stale_days=_env_int("NAV_STALE_DAYS", cls.nav_stale_days),
            candidate_k=_env_int("CANDIDATE_K", cls.candidate_k),
            top_k=_env_int("TOP_K", cls.top_k),
            weight_relevance=_env_float("WEIGHT_RELEVANCE", cls.weight_relevance),
            weight_returns=_env_float("WEIGHT_RETURNS", cls.weight_returns),
            weight_cost=_env_float("WEIGHT_COST", cls.weight_cost),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )
