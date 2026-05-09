"""Agent loop: sends messages, handles tool calls, repeats until done."""

import json
from typing import Any, Callable, Awaitable

from openai import OpenAI
from rich.console import Console

MAX_ITERATIONS = 15


async def run(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    call_tool: Callable[[str, dict], Awaitable[str]],
    console: Console,
    on_tool_call: Callable[[str, dict], None] | None = None,
) -> str:
    """
    Agent loop. Calls the model, executes tool calls, repeats.
    Returns the final assistant text.
    """
    for iteration in range(MAX_ITERATIONS):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # Append assistant turn to history
        msg_dict: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if choice.finish_reason != "tool_calls" or not msg.tool_calls:
            content = msg.content or ""
            console.print("\n[bold cyan]Assistant[/bold cyan]\n")
            print(content)
            return content

        # Execute all tool calls
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if on_tool_call:
                on_tool_call(tc.function.name, args)

            try:
                result = await call_tool(tc.function.name, args)
            except Exception as e:
                result = f"Tool error: {e}"

            # Truncate very long results to avoid context overflow
            if len(result) > 8000:
                result = result[:8000] + "\n...[truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Agent reached maximum iterations without a final answer."
