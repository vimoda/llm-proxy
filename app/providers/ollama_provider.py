import json
import uuid
from typing import AsyncGenerator, TypedDict, cast

import httpx

from app.config import settings
from app.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ModelInfo,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from app.providers.base import BaseProvider

BASE_URL = "http://localhost:11434/api"


class _OllamaToolCallFunction(TypedDict):
    name: str
    arguments: dict[str, object]


class _OllamaToolCall(TypedDict):
    function: _OllamaToolCallFunction


class _OllamaMessage(TypedDict, total=False):
    role: str
    content: str
    thinking: str
    tool_calls: list[_OllamaToolCall]
    images: list[str]


class _OllamaResponse(TypedDict, total=False):
    model: str
    created_at: str
    message: _OllamaMessage
    done: bool
    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class OllamaAIProvider(BaseProvider):
    # MODELS = ["llama3.2", "llama3.1", "mistral", "codellama", "phi3", "qwen2.5"]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.ollama_api_key}",
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
        if request.response_format is not None and request.response_format.type == "json_object":
            payload["format"] = "json"  # Ollama uses "format", not "response_format"
        return payload

    def _normalize(self, data: _OllamaResponse, model: str) -> ChatCompletionResponse:
        """Convert Ollama /api/chat response to OpenAI-compatible format."""
        message = data.get("message", _OllamaMessage(role="assistant", content=""))
        if not message:
            return ChatCompletionResponse(
                id=f"ollama-{uuid.uuid4().hex[:8]}",
                object="chat.completion",
                model=model,
                choices=[],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        # Convert Ollama tool_calls (no id, arguments as dict) → OpenAI format
        tool_calls: list[ToolCall] | None = None
        if ollama_tool_calls := message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=ToolCallFunction(
                        name=tc["function"]["name"],
                        arguments=json.dumps(tc["function"]["arguments"]),
                    ),
                )
                for tc in ollama_tool_calls
            ]

        content = message.get("content", "") or None  # empty string → None when tool_calls present

        return ChatCompletionResponse(
            id=f"ollama-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=data.get("done_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        payload = self._payload(request, model)
        payload["stream"] = False  # Ollama streams by default — disable it
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/chat",
                headers=self._headers(),
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            # Guardar la respuesta completa en un archivo para depuración
            # with open("ollama_response.json", "w") as f:
            #     json.dump(response.json(), f, indent=2)

            return self._normalize(cast(_OllamaResponse, response.json()), model)

    async def stream(  # type: ignore[override]
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(request, model)
        payload["stream"] = True
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/chat",
                headers=self._headers(),
                json=payload,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                # Ollama streams NDJSON — each line is a JSON object, not SSE.
                # We wrap it in SSE format for the client.
                async for line in response.aiter_lines():
                    if line:
                        yield f"data: {line}\n\n"

    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/tags",
                headers=self._headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # /api/tags → {"models": [{"name": "llama3.2:latest", "details": {...}}]}
        # /v1/models → {"data": [{"id": "llama3.2:latest"}]}
        native: list[dict[str, Any]] = data.get("models") or []
        oai: list[dict[str, Any]] = data.get("data") or []

        if native:
            return [
                ModelInfo(
                    id=m["name"],
                    name=m.get("name"),
                    description=m.get("details", {}).get("family"),
                )
                for m in native
            ]
        return [ModelInfo(id=m["id"]) for m in oai]
