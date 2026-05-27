import asyncio

from app.models import ModelInfo
from app.providers.base import BaseProvider
# from app.providers.openai_provider import OpenAIProvider
# from app.providers.anthropic_provider import AnthropicProvider
from app.providers.groq_provider import GroqAIProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.openrouter_provider import OpenRouterAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.ollama_provider import OllamaAIProvider
from app.providers.nvidia_provider import NvidiaProvider

_REGISTRY: dict[str, BaseProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "openrouter": OpenRouterAIProvider(),
    "ollama": OllamaAIProvider(),
    "groq": GroqAIProvider(),
    "nvidia": NvidiaProvider()
}


def list_provider_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_provider(name: str) -> BaseProvider:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


async def list_models() -> list[ModelInfo]:
    async def _fetch(provider_name: str, provider: BaseProvider) -> list[ModelInfo]:
        try:
            models = await provider.list_models()
        except Exception:
            models = [ModelInfo(id=m) for m in provider.MODELS]
        for m in models:
            m.id = f"{provider_name}/{m.id}"
            m.owned_by = provider_name
        return models

    results = await asyncio.gather(*[_fetch(name, p) for name, p in _REGISTRY.items()])
    all_models = [m for sublist in results for m in sublist]
    return sorted(all_models, key=lambda m: m.id.lower())
