from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.models import ChatCompletionRequest, ChatCompletionResponse, ModelInfo


class BaseProvider(ABC):
    MODELS: list[str] = []

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        """Return a full OpenAI-compatible chat completion response."""

    @abstractmethod
    def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[str, None]:
        """Yield SSE lines (e.g. 'data: {...}\\n\\n') until the stream ends."""

    async def list_models(self) -> list[ModelInfo]:
        """Return available models. Override to fetch dynamically from the provider API."""
        return [ModelInfo(id=model_id) for model_id in self.MODELS]
