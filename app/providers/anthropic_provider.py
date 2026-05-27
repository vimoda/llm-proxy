import json
from typing import Any, AsyncGenerator, TypedDict

import httpx

from app.config import settings
from app.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageUrlContentPart,
    Message,
    TextContentPart,
    Tool,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from app.providers.base import BaseProvider
from app.upstream import check_response, error_sse, UpstreamError

BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class _AnthropicUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class AnthropicProvider(BaseProvider):
    MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, object]]:
        """Convert OpenAI-format messages to Anthropic format."""
        result: list[dict[str, object]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "system":
                i += 1
                continue
            elif msg.role == "tool":
                # Group consecutive tool results into a single user message
                tool_results: list[dict[str, object]] = []
                while i < len(messages) and messages[i].role == "tool":
                    m = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content or "",
                    })
                    i += 1
                result.append({"role": "user", "content": tool_results})
            elif msg.role == "assistant" and msg.tool_calls:
                content: list[dict[str, object]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    })
                result.append({"role": "assistant", "content": content})
                i += 1
            else:
                result.append({"role": msg.role, "content": self._convert_content(msg)})
                i += 1
        return result

    def _convert_content(self, msg: Message) -> str | list[dict[str, object]]:
        """Convert content to Anthropic format (handles multimodal content parts)."""
        if not isinstance(msg.content, list):
            return msg.content or ""
        parts: list[dict[str, object]] = []
        for part in msg.content:
            if isinstance(part, TextContentPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageUrlContentPart):
                # Anthropic uses {"type": "image", "source": {"type": "url", "url": "..."}}
                parts.append({
                    "type": "image",
                    "source": {"type": "url", "url": part.image_url.url},
                })
        return parts

    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, object]]:
        return [
            {
                "name": t.function.name,
                "description": t.function.description or "",
                "input_schema": t.function.parameters or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

    def _convert_tool_choice(self, tool_choice: str | dict[str, Any] | None) -> dict[str, object]:
        if tool_choice is None or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function", {})
            assert isinstance(fn, dict)
            return {"type": "tool", "name": fn.get("name", "")}
        return {"type": "auto"}

    def _payload(self, request: ChatCompletionRequest, model: str) -> dict[str, object]:
        system = next(
            (m.content for m in request.messages if m.role == "system"), None
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": self._convert_messages(request.messages),
            "max_tokens": request.max_tokens or 1024,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)
            payload["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        return payload

    def _normalize(self, data: dict[str, object]) -> ChatCompletionResponse:
        """Convert Anthropic response to OpenAI-compatible format."""
        content_blocks = data.get("content", [])
        assert isinstance(content_blocks, list)
        usage: _AnthropicUsage = data["usage"]  # type: ignore[assignment]

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            assert isinstance(block, dict)
            if block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=str(block["id"]),
                    type="function",
                    function=ToolCallFunction(
                        name=str(block["name"]),
                        arguments=json.dumps(block.get("input", {})),
                    ),
                ))

        stop_reason = str(data.get("stop_reason", "stop"))
        finish_reason = "tool_calls" if stop_reason == "tool_use" else stop_reason

        return ChatCompletionResponse(
            id=str(data["id"]),
            object="chat.completion",
            model=str(data["model"]),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=" ".join(text_parts) if text_parts else None,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
                total_tokens=usage["input_tokens"] + usage["output_tokens"],
            ),
        )

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/messages",
                headers=self._headers(),
                json=self._payload(request, model),
                timeout=60.0,
            )
            await check_response(response)
            return self._normalize(response.json())

    async def stream(  # type: ignore[override]
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(request, model)
        payload["stream"] = True
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{BASE_URL}/messages",
                    headers=self._headers(),
                    json=payload,
                    timeout=60.0,
                ) as response:
                    await check_response(response)
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
            except UpstreamError as e:
                yield error_sse(e.status_code, str(e), e.body)
