"""Provider factory."""

from flagscale.agent.react.providers.base import LLMProvider


def get_provider(provider: str, model: str, api_key: str, base_url: str = None) -> LLMProvider:
    """Create an LLM provider instance."""
    if provider == "openai":
        from flagscale.agent.react.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model, api_key=api_key, base_url=base_url)
    elif provider == "anthropic":
        from flagscale.agent.react.providers.anthropic_provider import (
            AnthropicProvider,
        )

        return AnthropicProvider(model=model, api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")
