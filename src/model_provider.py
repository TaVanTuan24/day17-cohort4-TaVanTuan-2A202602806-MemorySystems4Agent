from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for chat model providers.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name and map common aliases/typos.

    Examples:
    - 'google' -> 'gemini'
    - 'anthorpic' -> 'anthropic'
    - strips whitespace and converts to lowercase
    """
    if not value or not isinstance(value, str):
        raise ValueError(f"Invalid provider: {value}")

    val = value.strip().lower()
    alias_map = {
        "google": "gemini",
        "google_genai": "gemini",
        "google-genai": "gemini",
        "anthorpic": "anthropic",
        "antropic": "anthropic",
        "claude": "anthropic",
        "open_ai": "openai",
        "open-ai": "openai",
        "open_router": "openrouter",
        "open-router": "openrouter",
        "local": "custom",
    }
    normalized = alias_map.get(val, val)

    valid_providers = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in valid_providers:
        raise ValueError(
            f"Unsupported provider: '{value}'. Must be one of {sorted(valid_providers)}"
        )
    return normalized


def build_chat_model(config: ProviderConfig):
    """Instantiate the chat model for the selected provider with lazy imports.

    Supported:
    - openai -> ChatOpenAI
    - custom -> ChatOpenAI with base_url
    - gemini -> ChatGoogleGenerativeAI
    - anthropic -> ChatAnthropic
    - ollama -> ChatOllama
    - openrouter -> ChatOpenRouter or ChatOpenAI with openrouter base_url
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "EMPTY",
            base_url=config.base_url,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )
    elif provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except (ImportError, Exception):
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or "EMPTY",
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
