"""Persistent key-value memory store for the agent."""

import json
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_PATH = Path.home() / ".llm-proxy" / "memory.json"


class MemoryStore:
    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------

    def save(self, key: str, content: str) -> str:
        now = datetime.now(timezone.utc).isoformat()
        existing = self._data.get(key)
        self._data[key] = {
            "content": content,
            "created": existing["created"] if existing else now,
            "updated": now,
        }
        self._save()
        return f"Memory '{key}' saved."

    def recall(self, key: str) -> str:
        entry = self._data.get(key)
        if not entry:
            return f"No memory found for '{key}'."
        return f"{key}: {entry['content']}  [updated: {entry['updated']}]"

    def list_all(self) -> str:
        if not self._data:
            return "No memories stored."
        lines = [f"- {k}: {v['content']}" for k, v in self._data.items()]
        return "\n".join(lines)

    def delete(self, key: str) -> str:
        if key not in self._data:
            return f"No memory found for '{key}'."
        del self._data[key]
        self._save()
        return f"Memory '{key}' deleted."

    def context_block(self) -> str:
        """Format all memories for injection into the system prompt."""
        if not self._data:
            return ""
        lines = [f"- {k}: {v['content']}" for k, v in self._data.items()]
        return "## Memories (from previous sessions)\n" + "\n".join(lines)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Tool schemas + handlers
# ---------------------------------------------------------------------------

MEMORY_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save or update a named memory that will persist across conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short identifier for this memory (e.g. 'user_name', 'preferred_language')."},
                    "content": {"type": "string", "description": "The information to remember."},
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Retrieve a specific memory by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The memory key to look up."},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List all stored memories.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Delete a memory by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The memory key to delete."},
                },
                "required": ["key"],
            },
        },
    },
]


def make_memory_handlers(store: MemoryStore) -> dict[str, callable]:
    return {
        "save_memory": lambda args: store.save(args["key"], args["content"]),
        "recall_memory": lambda args: store.recall(args["key"]),
        "list_memories": lambda _args: store.list_all(),
        "delete_memory": lambda args: store.delete(args["key"]),
    }
