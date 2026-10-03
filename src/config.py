from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def resolve_api_key(provider: str, override_key: str | None = None) -> str | None:
    """Resolve API key specifically for a given provider with explicit override support."""
    if override_key:
        return override_key

    try:
        norm = normalize_provider(provider)
    except ValueError:
        norm = provider.strip().lower()

    key_map = {
        "openai": os.environ.get("OPENAI_API_KEY"),
        "gemini": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "openrouter": os.environ.get("OPENROUTER_API_KEY"),
        "custom": os.environ.get("CUSTOM_API_KEY"),
        "ollama": None,
    }
    return key_map.get(norm)


def resolve_base_url(provider: str) -> str | None:
    """Resolve base URL specifically for a given provider."""
    try:
        norm = normalize_provider(provider)
    except ValueError:
        norm = provider.strip().lower()

    if norm == "custom":
        return os.environ.get("CUSTOM_BASE_URL")
    elif norm == "ollama":
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    elif norm == "openrouter":
        return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return None


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load configuration from environment variables or sensible defaults."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    env_path = root / ".env"
    if env_path.exists():
        try:
            import dotenv

            dotenv.load_dotenv(env_path)
        except ImportError:
            pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"

    compact_threshold = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", 800))
    compact_keep = int(os.environ.get("COMPACT_KEEP_MESSAGES", 4))

    # Main model configuration
    provider = os.environ.get("LLM_PROVIDER", "custom")
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    llm_api_key_override = os.environ.get("LLM_API_KEY")
    api_key = resolve_api_key(provider, override_key=llm_api_key_override)
    base_url = resolve_base_url(provider)

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge model configuration (isolated resolution)
    judge_provider = os.environ.get("JUDGE_PROVIDER", provider)
    judge_model_name = os.environ.get("JUDGE_MODEL", model_name)
    judge_api_key_override = os.environ.get("JUDGE_API_KEY")
    judge_api_key = resolve_api_key(judge_provider, override_key=judge_api_key_override)
    judge_base_url = resolve_base_url(judge_provider)

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config,
    )
