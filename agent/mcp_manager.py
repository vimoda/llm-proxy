"""MCP server manager — loads servers from mcp.json and exposes their tools."""

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any


class MCPManager:
    def __init__(self, config_path: str = "mcp.json") -> None:
        self.config_path = Path(config_path)
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}  # full_name → (server, tool)
        self._tools: list[dict] = []

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {"mcpServers": {}}
        with open(self.config_path) as f:
            return json.load(f)

    async def start(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return

        config = self._load_config()
        for name, srv in config.get("mcpServers", {}).items():
            try:
                params = StdioServerParameters(
                    command=srv["command"],
                    args=srv.get("args", []),
                    env={**os.environ, **{k: os.path.expandvars(v) for k, v in srv.get("env", {}).items()}},
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session

                result = await session.list_tools()
                for tool in result.tools:
                    full_name = f"{name}__{tool.name}"
                    self._tool_map[full_name] = (name, tool.name)
                    self._tools.append({
                        "type": "function",
                        "function": {
                            "name": full_name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    })
            except Exception as e:
                print(f"[MCP] Failed to start server '{name}': {e}")

    async def call_tool(self, full_name: str, args: dict) -> str:
        server_name, tool_name = self._tool_map[full_name]
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, args)
        parts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(parts) or "(no output)"

    def is_mcp_tool(self, name: str) -> bool:
        return name in self._tool_map

    @property
    def tools(self) -> list[dict]:
        return self._tools

    @property
    def server_count(self) -> int:
        return len(self._sessions)

    async def stop(self) -> None:
        await self._stack.aclose()
