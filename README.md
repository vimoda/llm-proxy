# LLM Proxy

OpenAI-compatible HTTP proxy that routes requests to multiple LLM providers. Drop-in replacement for the OpenAI API — point any OpenAI SDK at `http://localhost:8000/v1`.

## Providers

| Provider | Models endpoint | Notes |
|----------|----------------|-------|
| `openrouter` | Dynamic (fetches from API, free models only) | Multimodal, tool calling |
| `ollama` | Static (`MODELS` list) | Local inference |
| `groq` | Dynamic (fetches from API) | Tool calling |
| `nvidia` | Static (`MODELS` list) | Tool calling |

## Quick start

Use a virtual environment in the repo (for example `.venv`) so dependencies stay isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python main.py
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration

```env
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
NVIDIA_API_KEY=...
OLLAMA_API_KEY=...          # optional
API_KEYS=key-v1,key-v2      # omit to disable auth
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RPM=60
RATE_LIMIT_TPM=100000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (no auth required) |
| GET | `/v1/providers` | List active providers |
| GET | `/v1/models` | List all models across providers |
| GET | `/v1/models/{provider}` | List models for one provider |
| POST | `/v1/chat/completions` | Chat completion (streaming supported) |

## Proxy examples

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### List providers

```bash
curl http://localhost:8000/v1/providers \
  -H "Authorization: Bearer my-key"
```
```json
{"object":"list","data":[{"id":"openrouter"},{"id":"ollama"},{"id":"groq"},{"id":"nvidia"}]}
```

### List models for a provider

```bash
curl http://localhost:8000/v1/models/groq \
  -H "Authorization: Bearer my-key"
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

### System prompt

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b-instant",
    "messages": [
      {"role": "system", "content": "Reply only in Spanish."},
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

### Multi-turn conversation

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b-instant",
    "messages": [
      {"role": "user", "content": "My name is Daniel."},
      {"role": "assistant", "content": "Nice to meet you, Daniel!"},
      {"role": "user", "content": "What is my name?"}
    ]
  }'
```

### Tool calling

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-70b-versatile",
    "messages": [{"role": "user", "content": "What is the weather in Madrid?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Madrid\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Multimodal (image input)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openrouter/google/gemini-flash-1.5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
      ]
    }]
  }'
```

### JSON mode

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Return a JSON object with fields: name, age, city. Use fake data."}],
    "response_format": {"type": "json_object"}
  }'
```

### Temperature and max tokens

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "temperature": 0.9,
    "max_tokens": 60
  }'
```

## CLI client

`client.py` — Python CLI built on the OpenAI SDK (`openai>=1.0.0`, `rich>=13.0.0`).

Configure via `.env` or env vars:

```env
LLM_PROXY_URL=http://localhost:8000/v1
LLM_PROXY_KEY=my-key        # omit if auth disabled
```

### Interactive UI

```bash
python client.py       # or: python client.py ui
```

Displays a numbered model list on startup, then enters a chat REPL with streaming output.

| Command | Description |
|---------|-------------|
| `/model` | Pick a different model |
| `/system [msg]` | Set system prompt (blank to clear) |
| `/new` | Clear conversation history |
| `/history` | Show full conversation history |
| `/help` | Show available commands |
| `/exit` | Quit |

### Agent UI (tool use + MCP)

```bash
python client.py agent
```

Agent mode picks a model, then runs an **agentic loop**: the model can call tools, results feed back into context, and the loop repeats until a final answer (max 15 iterations).

**Built-in tools** (always available unless `--no-builtin`):

| Tool | Description |
|------|-------------|
| `read_file` | Read a file from the filesystem |
| `list_directory` | List files in a directory |
| `run_shell` | Execute a shell command |

**Memory tools** (always available — persisted to `~/.llm-proxy/memory.json`):

| Tool | Description |
|------|-------------|
| `save_memory` | Store a named fact across sessions |
| `recall_memory` | Retrieve a specific memory by key |
| `list_memories` | List all stored memories |
| `delete_memory` | Delete a memory by key |

All memories are automatically injected into the system prompt at the start of every turn — the model remembers without needing to call tools explicitly.

**MCP tools** — loaded from `mcp.json` at startup:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

Each MCP server's tools are exposed as `servername__toolname`. Multiple servers can be listed under `"servers"`.

```bash
# With default mcp.json
python client.py agent

# Custom MCP config + system prompt
python client.py agent --mcp ~/.config/mcp.json --system "You are a coding assistant"

# Built-in tools only, no MCP
python client.py agent --no-builtin
```

| Command | Description |
|---------|-------------|
| `/model` | Pick a different model |
| `/system [msg]` | Set system prompt |
| `/tools` | List all available tools |
| `/memory` | Show all stored memories |
| `/forget <key>` | Delete a memory entry |
| `/new` | Clear conversation history |
| `/help` | Show available commands |
| `/exit` | Quit |

### Non-interactive commands

```bash
# List providers
python client.py providers

# List models (all or by provider)
python client.py models
python client.py models --provider groq

# Single-shot chat
python client.py chat groq/llama-3.1-8b-instant "What is a proxy?"
python client.py chat openrouter/google/gemini-flash-1.5 "Hello" \
  --system "Always reply in Spanish" \
  --temperature 0.5 --max-tokens 500

# Streaming
python client.py stream groq/llama-3.1-8b-instant "Tell me a joke"
```

## Documentation

See **[docs/usage.md](docs/usage.md)** for full client examples:
- Shell (curl)
- Python — raw HTTP
- Python — OpenAI SDK
- Node.js — raw HTTP
- Node.js — OpenAI SDK

## Development

```bash
pytest                                                      # all tests
pytest tests/test_api.py::test_anthropic_complete_normalizes_response -v  # single test
```

Adding a provider: implement `BaseProvider` in `app/providers/<name>_provider.py`, register an instance in `app/providers/__init__.py`. See [CLAUDE.md](CLAUDE.md) for architecture details.
