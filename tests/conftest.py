import pytest
from fastapi.testclient import TestClient

from app.config import settings
from main import app


@pytest.fixture(autouse=True)
def _no_auth():
    """Disable API key auth for tests."""
    saved = settings.api_keys
    settings.api_keys = ""
    yield
    settings.api_keys = saved


@pytest.fixture
def client():
    return TestClient(app)


# Minimal valid request body
BASE_REQUEST = {
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
}

# Shared fake OpenAI response
OPENAI_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

# Shared fake Anthropic response
ANTHROPIC_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Hi there!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
