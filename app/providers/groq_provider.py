from typing import AsyncGenerator, TypedDict, cast

import httpx

from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
)
from app.providers.base import BaseProvider

BASE_URL = "https://api.groq.com/openai/v1"


class _GroqModel(TypedDict, total=False):
    id: str
    object: str
    created: int
    owned_by: str
    active: bool
    context_window: int
    max_completion_tokens: int


class GroqAIProvider(BaseProvider):

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.groq_api_key}",
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
            response.raise_for_status()
            return ChatCompletionResponse.model_validate(response.json())

    def stream(  # type: ignore[override]
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[str, None]:
        async def _inner() -> AsyncGenerator[str, None]:
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
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
        return _inner()

    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/models",
                headers=self._headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        result: list[ModelInfo] = []
        for raw in data.get("data", []):
            m = cast(_GroqModel, raw)
            if not m.get("active", True):
                continue
            result.append(ModelInfo(
                id=m.get("id", ""),
                created=m.get("created"),
                name=m.get("id"),
                context_length=m.get("context_window"),
                max_completion_tokens=m.get("max_completion_tokens"),
                owned_by=m.get("owned_by", ""),
            ))
        return result
