# LLM Proxy — Usage Guide

Base URL: `http://localhost:8000`  
Auth: `Authorization: Bearer <key>` (omit if `API_KEYS` not set)  
Model format: `provider/model` — e.g. `openrouter/google/gemini-flash-1.5`

---

## Shell (curl)

### List providers
```bash
curl http://localhost:8000/v1/providers \
  -H "Authorization: Bearer my-key"
```

### List models
```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer my-key"
```

### Chat completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openrouter/google/gemini-flash-1.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer my-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/minimaxai/minimax-m2.7",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## Python — raw HTTP (`httpx`)

```python
import httpx

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer my-key"}


def list_models():
    r = httpx.get(f"{BASE_URL}/v1/models", headers=HEADERS)
    r.raise_for_status()
    return r.json()["data"]


def chat(model: str, messages: list[dict]) -> str:
    r = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json={"model": model, "messages": messages},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_stream(model: str, messages: list[dict]):
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json={"model": model, "messages": messages, "stream": True},
        timeout=60,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                import json
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)


# Usage
print(chat("openrouter/google/gemini-flash-1.5", [{"role": "user", "content": "Hello!"}]))
chat_stream("groq/llama-3.1-8b-instant", [{"role": "user", "content": "Tell me a joke"}])
```

---

## Python — OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-key",  # any non-empty string if auth disabled
)


def chat(model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def chat_stream(model: str, messages: list[dict]):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


def chat_with_tools(model: str, messages: list[dict], tools: list[dict]):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0]


# Usage
reply = chat(
    "openrouter/google/gemini-flash-1.5",
    [{"role": "user", "content": "Hello!"}],
)
print(reply)

chat_stream(
    "groq/llama-3.1-8b-instant",
    [{"role": "user", "content": "Tell me a joke"}],
)

# Tool calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
choice = chat_with_tools(
    "openrouter/openai/gpt-4o",
    [{"role": "user", "content": "What's the weather in Madrid?"}],
    tools,
)
if choice.finish_reason == "tool_calls":
    print(choice.message.tool_calls[0].function.arguments)
```

---

## Node.js — raw HTTP (`fetch`)

```js
const BASE_URL = "http://localhost:8000";
const HEADERS = {
  Authorization: "Bearer my-key",
  "Content-Type": "application/json",
};

async function listModels() {
  const res = await fetch(`${BASE_URL}/v1/models`, { headers: HEADERS });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.data;
}

async function chat(model, messages) {
  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ model, messages }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.choices[0].message.content;
}

async function chatStream(model, messages) {
  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ model, messages, stream: true }),
  });
  if (!res.ok) throw new Error(await res.text());

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    for (const line of text.split("\n")) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        const chunk = JSON.parse(line.slice(6));
        const delta = chunk.choices[0].delta?.content ?? "";
        process.stdout.write(delta);
      }
    }
  }
  console.log();
}

// Usage
const reply = await chat("openrouter/google/gemini-flash-1.5", [
  { role: "user", content: "Hello!" },
]);
console.log(reply);

await chatStream("groq/llama-3.1-8b-instant", [
  { role: "user", content: "Tell me a joke" },
]);
```

---

## Node.js — OpenAI SDK

```js
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "my-key", // any non-empty string if auth disabled
});

async function chat(model, messages) {
  const response = await client.chat.completions.create({ model, messages });
  return response.choices[0].message.content;
}

async function chatStream(model, messages) {
  const stream = await client.chat.completions.create({
    model,
    messages,
    stream: true,
  });
  for await (const chunk of stream) {
    const delta = chunk.choices[0].delta?.content ?? "";
    process.stdout.write(delta);
  }
  console.log();
}

async function chatWithTools(model, messages, tools) {
  const response = await client.chat.completions.create({
    model,
    messages,
    tools,
    tool_choice: "auto",
  });
  return response.choices[0];
}

// Usage
const reply = await chat("openrouter/google/gemini-flash-1.5", [
  { role: "user", content: "Hello!" },
]);
console.log(reply);

await chatStream("groq/llama-3.1-8b-instant", [
  { role: "user", content: "Tell me a joke" },
]);

// Tool calling
const tools = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get weather for a city",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
    },
  },
];
const choice = await chatWithTools(
  "openrouter/openai/gpt-4o",
  [{ role: "user", content: "What's the weather in Madrid?" }],
  tools
);
if (choice.finish_reason === "tool_calls") {
  console.log(choice.message.tool_calls[0].function.arguments);
}
```

---

## Error responses

All errors return OpenAI-compatible JSON:

```json
{
  "error": {
    "message": "Rate limit exceeded. Retry after 60s.",
    "type": "rate_limit_error",
    "code": "429"
  }
}
```

| Status | Cause |
|--------|-------|
| 401 | Missing or invalid API key |
| 400 | Bad model format or unknown provider |
| 422 | Invalid request body |
| 429 | RPM or TPM rate limit exceeded |
