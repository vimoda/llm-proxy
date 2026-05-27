"""Tests for POST /v1/chat/completions — routing, validation, streaming."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import BASE_REQUEST, OPENAI_RESPONSE, ANTHROPIC_RESPONSE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_missing_provider_separator(client):
    resp = client.post("/v1/chat/completions", json={**BASE_REQUEST, "model": "gpt-4o"})
    assert resp.status_code == 400
    assert "provider/model" in resp.json()["error"]["message"]


def test_unknown_provider(client):
    resp = client.post("/v1/chat/completions", json={**BASE_REQUEST, "model": "unknown/model"})
    assert resp.status_code == 400
    assert "unknown" in resp.json()["error"]["message"].lower()


def test_invalid_role(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "bot", "content": "Hi"}]},
    )
    assert resp.status_code == 422


def test_empty_messages(client):
    resp = client.post("/v1/chat/completions", json={"model": "openai/gpt-4o", "messages": []})
    # FastAPI accepts empty list — provider will fail, but schema itself is valid
    # We just verify it does not 422
    assert resp.status_code != 422


# ---------------------------------------------------------------------------
# OpenAI — complete
# ---------------------------------------------------------------------------

def _ok_response(data: dict) -> MagicMock:
    m = MagicMock()
    m.json.return_value = data
    m.is_error = False
    return m


def test_openai_complete(client):
    mock_response = _ok_response(OPENAI_RESPONSE)

    with patch(
        "app.providers.openai_provider.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        resp = client.post("/v1/chat/completions", json=BASE_REQUEST)

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "chatcmpl-123"
    assert body["choices"][0]["message"]["content"] == "Hi there!"


def test_openai_passes_temperature_and_max_tokens(client):
    mock_response = _ok_response(OPENAI_RESPONSE)

    with patch(
        "app.providers.openai_provider.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        client.post(
            "/v1/chat/completions",
            json={**BASE_REQUEST, "temperature": 0.5, "max_tokens": 100},
        )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["temperature"] == 0.5
    assert call_kwargs["json"]["max_tokens"] == 100


def test_openai_omits_optional_fields_when_none(client):
    mock_response = _ok_response(OPENAI_RESPONSE)

    with patch(
        "app.providers.openai_provider.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        client.post("/v1/chat/completions", json=BASE_REQUEST)

    call_kwargs = mock_post.call_args.kwargs
    assert "temperature" not in call_kwargs["json"]
    assert "max_tokens" not in call_kwargs["json"]


# ---------------------------------------------------------------------------
# OpenAI — stream
# ---------------------------------------------------------------------------

def test_openai_stream(client):
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        "data: [DONE]",
    ]

    async def fake_aiter_lines():
        for line in sse_lines:
            yield line

    mock_stream_response = MagicMock()
    mock_stream_response.is_error = False
    mock_stream_response.aiter_lines = fake_aiter_lines
    mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
    mock_stream_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_stream_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.providers.openai_provider.httpx.AsyncClient", return_value=mock_client):
        resp = client.post(
            "/v1/chat/completions",
            json={**BASE_REQUEST, "stream": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    content = resp.text
    assert 'data: {"choices":[{"delta":{"content":"Hi"}}]}' in content
    assert "data: [DONE]" in content


# ---------------------------------------------------------------------------
# Anthropic — complete (normalization)
# ---------------------------------------------------------------------------

def test_anthropic_complete_normalizes_response(client):
    mock_response = _ok_response(ANTHROPIC_RESPONSE)

    with patch(
        "app.providers.anthropic_provider.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={**BASE_REQUEST, "model": "anthropic/claude-3-5-sonnet-20241022"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hi there!"
    assert body["usage"]["prompt_tokens"] == 5
    assert body["usage"]["completion_tokens"] == 3
    assert body["usage"]["total_tokens"] == 8


def test_anthropic_extracts_system_message(client):
    mock_response = _ok_response(ANTHROPIC_RESPONSE)

    with patch(
        "app.providers.anthropic_provider.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["system"] == "Be concise."
    # system message must not appear in messages array
    assert all(m["role"] != "system" for m in payload["messages"])


def test_anthropic_default_max_tokens(client):
    mock_response = _ok_response(ANTHROPIC_RESPONSE)

    with patch(
        "app.providers.anthropic_provider.httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        client.post(
            "/v1/chat/completions",
            json={**BASE_REQUEST, "model": "anthropic/claude-3-5-sonnet-20241022"},
        )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 1024


# ---------------------------------------------------------------------------
# Anthropic — stream
# ---------------------------------------------------------------------------

def test_anthropic_stream(client):
    sse_lines = [
        'data: {"type":"content_block_delta","delta":{"text":"Hi"}}',
        'data: {"type":"message_stop"}',
    ]

    async def fake_aiter_lines():
        for line in sse_lines:
            yield line

    mock_stream_response = MagicMock()
    mock_stream_response.is_error = False
    mock_stream_response.aiter_lines = fake_aiter_lines
    mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
    mock_stream_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_stream_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.providers.anthropic_provider.httpx.AsyncClient", return_value=mock_client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert 'data: {"type":"content_block_delta"' in resp.text
