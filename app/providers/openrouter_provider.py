import json
from typing import AsyncGenerator, TypedDict, cast

import httpx

from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelArchitecture,
    ModelInfo,
    ModelPricing,
)
from app.providers.base import BaseProvider


class _ORArchitecture(TypedDict, total=False):
    modality: str
    input_modalities: list[str]
    output_modalities: list[str]
    tokenizer: str


class _ORPricing(TypedDict, total=False):
    prompt: str
    completion: str
    input_cache_read: str


class _ORTopProvider(TypedDict, total=False):
    context_length: int
    max_completion_tokens: int
    is_moderated: bool


class _ORModel(TypedDict, total=False):
    id: str
    created: int
    name: str
    description: str
    context_length: int
    architecture: _ORArchitecture
    pricing: _ORPricing
    top_provider: _ORTopProvider
    supported_parameters: list[str]

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAIProvider(BaseProvider):
    # MODELS = [
    #     "meta-llama/llama-3.1-8b-instruct",
    #     "meta-llama/llama-3.1-70b-instruct",
    #     "google/gemini-flash-1.5",
    #     "google/gemini-pro-1.5",
    #     "mistralai/mistral-7b-instruct",
    #     "mistralai/mixtral-8x7b-instruct",
    #     "deepseek/deepseek-chat",
    # ]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "localhost-llm-proxy.com",
            "X-OpenRouter-Title": "LLM Proxy",
        }

    def _payload(self, request: ChatCompletionRequest, model: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [t.model_dump() for t in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            payload["response_format"] = request.response_format.model_dump()
        return payload

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers=self._headers(),
                json=self._payload(request, model),
                timeout=60.0,
            )

            # Imprimir estatus de la respuesta para depuración
            print(f"OpenRouter response status: {response.status_code}")
            response.raise_for_status()
            # Guardar la respuesta completa en un archivo para depuración
            # with open("openrouter_response.json", "w") as f:
            #     json.dump(response.json(), f, indent=2)

            return ChatCompletionResponse.model_validate(response.json())

    async def stream(  # type: ignore[override]
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(request, model)
        payload["stream"] = True
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    print(f"OpenRouter stream line: {line}")  # Debug: print each line received
                    if line.startswith("data: "):
                        yield f"{line}\n\n"

    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/models",
                headers=self._headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            # Guardar la respuesta completa en un archivo para depuración
            # with open("openrouter_models_response.json", "w") as f:
            #     json.dump(data, f, indent=2)

            result: list[ModelInfo] = []
            for raw in data.get("data", []):
                m = cast(_ORModel, raw)
                arch = m.get("architecture") or cast(_ORArchitecture, {})
                pricing = m.get("pricing") or cast(_ORPricing, {})
                top = m.get("top_provider") or cast(_ORTopProvider, {})
                is_free = pricing.get("prompt") == "0" and pricing.get("completion") == "0"
                if not is_free:
                    continue  # Skip paid models for now
                
                result.append(ModelInfo(
                    id=m.get("id", ""),
                    is_free=is_free,
                    created=m.get("created"),
                    name=m.get("name"),
                    description=m.get("description"),
                    context_length=m.get("context_length"),
                    max_completion_tokens=top.get("max_completion_tokens"),
                    architecture=ModelArchitecture(
                        modality=arch.get("modality"),
                        input_modalities=arch.get("input_modalities") or [],
                        output_modalities=arch.get("output_modalities") or [],
                        tokenizer=arch.get("tokenizer"),
                    ) if arch else None,
                    pricing=ModelPricing(
                        prompt=pricing.get("prompt"),
                        completion=pricing.get("completion"),
                        input_cache_read=pricing.get("input_cache_read"),
                    ) if pricing else None,
                    supported_parameters=m.get("supported_parameters") or [],
                ))
            return result
