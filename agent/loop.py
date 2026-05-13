"""Agent loop: sends messages, handles tool calls, repeats until done."""

import json
from typing import Any, Callable, Awaitable

from openai import OpenAI
from rich.console import Console

MAX_ITERATIONS = 15


def _stream_one_assistant_turn(
    client: OpenAI,
    kwargs: dict[str, Any],
    console: Console,
) -> tuple[dict[str, Any], str | None]:
    """
    Stream one chat completion. Prints assistant output incrementally.
    Returns (assistant_message_dict for history, finish_reason).
    """
    kwargs = {**kwargs, "stream": True}
    stream = client.chat.completions.create(**kwargs)

    content_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None

    status = console.status("[dim]Assistant is thinking…[/dim]", spinner="dots")
    status.start()
    thinking = True
    header_shown = False
    reasoning_label_shown = False
    content_started = False

    def stop_thinking() -> None:
        nonlocal thinking
        if thinking:
            status.stop()
            thinking = False

    def ensure_header() -> None:
        nonlocal header_shown
        stop_thinking()
        if not header_shown:
            console.print("\n[bold cyan]Assistant[/bold cyan]\n")
            header_shown = True

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            c0 = chunk.choices[0]
            if c0.finish_reason is not None:
                finish_reason = c0.finish_reason
            d = c0.delta

            d_reason = getattr(d, "reasoning_content", None) or ""
            if d_reason:
                ensure_header()
                if not reasoning_label_shown:
                    console.print("[dim italic]Reasoning[/dim italic]\n", end="")
                    reasoning_label_shown = True
                console.print(d_reason, end="", style="dim", markup=False)

            d_content = d.content or ""
            if d_content:
                ensure_header()
                if reasoning_label_shown and not content_started:
                    console.print()
                    content_started = True
                print(d_content, end="", flush=True)
                content_parts.append(d_content)
                content_started = True

            for tc in getattr(d, "tool_calls", None) or []:
                stop_thinking()
                idx = getattr(tc, "index", None)
                if idx is None:
                    idx = 0
                if idx not in tool_acc:
                    tool_acc[idx] = {"id": "", "name": "", "arguments": ""}
                if getattr(tc, "id", None):
                    tool_acc[idx]["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", None) or ""
                    if name:
                        tool_acc[idx]["name"] += name
                    args = getattr(fn, "arguments", None) or ""
                    if args:
                        tool_acc[idx]["arguments"] += args
    finally:
        stop_thinking()

    full_content = "".join(content_parts)
    tool_calls: list[dict[str, Any]] = []
    for i in sorted(tool_acc.keys()):
        t = tool_acc[i]
        tool_calls.append({
            "id": t["id"],
            "type": "function",
            "function": {"name": t["name"], "arguments": t["arguments"]},
        })

    if header_shown or full_content:
        console.print()
    elif tool_calls:
        console.print()

    msg_dict: dict[str, Any] = {"role": "assistant", "content": full_content or None}
    if tool_calls:
        msg_dict["tool_calls"] = tool_calls

    return msg_dict, finish_reason


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
    for _iteration in range(MAX_ITERATIONS):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        msg_dict, finish_reason = _stream_one_assistant_turn(client, kwargs, console)
        messages.append(msg_dict)

        tool_calls = msg_dict.get("tool_calls") or []
        if finish_reason != "tool_calls" or not tool_calls:
            return (msg_dict.get("content") or "") or ""

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            args = json.loads(fn.get("arguments") or "{}")
            if on_tool_call:
                on_tool_call(name, args)

            try:
                result = await call_tool(name, args)
            except Exception as e:
                result = f"Tool error: {e}"

            if len(result) > 8000:
                result = result[:8000] + "\n...[truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return "Agent reached maximum iterations without a final answer."
