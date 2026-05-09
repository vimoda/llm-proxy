"""
LLM Proxy CLI client — requires: openai>=1.0.0, rich>=13.0.0, mcp>=1.0.0
Usage:
    python client.py                                         # interactive UI
    python client.py agent [--mcp mcp.json] [--no-builtin]  # agent UI with tools
    python client.py providers
    python client.py models [--provider <name>]
    python client.py chat <model> <message> [--system <msg>] [--temperature 0.7] [--max-tokens 1024]
    python client.py stream <model> <message> [--system <msg>] [--temperature 0.7] [--max-tokens 1024]
"""

import argparse
import asyncio
import os
from re import A
import readline  # noqa: F401 — enables input history/editing
import sys

from dotenv import load_dotenv
from openai import OpenAI
import httpx
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

load_dotenv()

BASE_URL = os.environ.get("LLM_PROXY_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("LLM_PROXY_KEY", "no-key")

console = Console()

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/model[/cyan]          Change model
  [cyan]/system [msg][/cyan]   Set system prompt (blank to clear)
  [cyan]/new[/cyan]            Clear conversation history
  [cyan]/history[/cyan]        Show conversation history
  [cyan]/help[/cyan]           Show this help
  [cyan]/exit[/cyan]           Quit
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def _raw_get(path: str) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = httpx.get(f"{BASE_URL}{path}", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _pick_model() -> str:
    console.print("\n[bold]Fetching models...[/bold]")
    try:
        data = _raw_get("/models")
    except Exception as e:
        console.print(f"[red]Could not fetch models: {e}[/red]")
        return Prompt.ask("[bold]Enter model manually[/bold]")

    models = data.get("data", [])
    if not models:
        console.print("[yellow]No models returned by proxy.[/yellow]")
        return Prompt.ask("[bold]Enter model manually[/bold]")

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=4)
    table.add_column("Model ID", style="green")
    table.add_column("Name")
    table.add_column("Context", justify="right", style="dim")
    table.add_column("", style="yellow")

    for i, m in enumerate(models, 1):
        ctx = f"{m['context_length']:,}" if m.get("context_length") else ""
        free = "free" if m.get("is_free") else ""
        table.add_row(str(i), m["id"], m.get("name") or "", ctx, free)

    console.print(table)

    while True:
        choice = Prompt.ask(f"\n[bold]Pick model[/bold] [dim](1-{len(models)} or type id)[/dim]")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]["id"]
            console.print("[red]Out of range.[/red]")
        elif "/" in choice:
            return choice
        else:
            console.print("[red]Enter a number or provider/model.[/red]")


def _stream_and_capture(
    model: str,
    messages: list[dict],
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    kwargs: dict = {"model": model, "messages": messages, "stream": True}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    console.print("\n[bold cyan]Assistant[/bold cyan]\n")
    full_text = ""
    stream = _client().chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            print(delta, end="", flush=True)
            full_text += delta
    print()
    return full_text


# ---------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------

def cmd_ui(_args: argparse.Namespace | None = None) -> None:
    console.print(
        Panel.fit(
            "[bold]LLM Proxy[/bold] interactive client\n"
            "[dim]Type [cyan]/help[/cyan] for commands · [cyan]/exit[/cyan] to quit[/dim]",
            border_style="cyan",
        )
    )

    model = _pick_model()
    system_prompt: str | None = None
    history: list[dict] = []
    temperature: float | None = None
    max_tokens: int | None = None

    def _status() -> str:
        sys_tag = " [dim]+ system[/dim]" if system_prompt else ""
        turns = len([m for m in history if m["role"] == "user"])
        return (
            f"[bold green]{model}[/bold green]{sys_tag} "
            f"[dim]· {turns} turn{'s' if turns != 1 else ''}[/dim]"
        )

    console.print(f"\nUsing: {_status()}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            rest = user_input[len(cmd):].strip()

            if cmd == "/exit":
                console.print("[dim]Bye.[/dim]")
                break
            elif cmd == "/new":
                history.clear()
                console.print("[dim]Conversation cleared.[/dim]")
            elif cmd == "/model":
                model = _pick_model()
                console.print(f"\nUsing: {_status()}\n")
            elif cmd == "/system":
                if rest:
                    system_prompt = rest
                else:
                    typed = Prompt.ask("[bold]System prompt[/bold] [dim](blank to clear)[/dim]")
                    system_prompt = typed or None
                label = f"[green]{system_prompt}[/green]" if system_prompt else "[dim]cleared[/dim]"
                console.print(f"System prompt: {label}")
            elif cmd == "/history":
                if not history:
                    console.print("[dim]No history.[/dim]")
                else:
                    for m in history:
                        color = "cyan" if m["role"] == "assistant" else "white"
                        console.print(f"[{color}]{m['role'].capitalize()}:[/{color}] {m['content']}")
            elif cmd == "/help":
                rprint(HELP_TEXT)
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]  Type [cyan]/help[/cyan].")
            continue

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        try:
            reply = _stream_and_capture(model, messages, temperature, max_tokens)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
        except httpx.HTTPStatusError as e:
            console.print(f"[red]HTTP {e.response.status_code}:[/red] {e.response.text}")
            continue
        except Exception as e:
            console.print(f"[red]{type(e).__name__}:[/red] {e!r}")
            continue

        console.print(f"\n[dim]{_status()}[/dim]\n")


# ---------------------------------------------------------------------------
# Agent UI
# ---------------------------------------------------------------------------

def cmd_agent(args: argparse.Namespace) -> None:
    asyncio.run(_agent_ui(args))


async def _agent_ui(args: argparse.Namespace) -> None:
    from agent.mcp_manager import MCPManager
    from agent.memory import MemoryStore, MEMORY_TOOLS, make_memory_handlers
    from agent.tools import BUILTIN_TOOLS, BUILTIN_HANDLERS
    import agent.loop as agent_loop

    console.print(
        Panel.fit(
            "[bold]LLM Proxy[/bold] agent client\n"
            "[dim]Models can use tools + memory. Type [cyan]/help[/cyan] for commands · [cyan]/exit[/cyan] to quit[/dim]",
            border_style="yellow",
        )
    )

    # Memory
    memory = MemoryStore()
    memory_handlers = make_memory_handlers(memory)
    console.print(f"[dim]Memory: {len(memory)} stored entries (in {memory.path})[/dim]")

    # MCP
    mcp = MCPManager(config_path=args.mcp)
    console.print(f"[dim]Loading MCP servers from {args.mcp}...[/dim]")
    await mcp.start()
    if mcp.server_count:
        console.print(f"[green]✓ {mcp.server_count} MCP server(s) · {len(mcp.tools)} tool(s)[/green]")
    else:
        console.print("[dim]No MCP servers loaded.[/dim]")

    # Assemble tools
    all_tools: list[dict] = [*MEMORY_TOOLS]
    if not args.no_builtin:
        all_tools.extend(BUILTIN_TOOLS)
    all_tools.extend(mcp.tools)
    # Remove the shell tool from the exposed tool list.
    all_tools = [
        t
        for t in all_tools
        if not (isinstance(t.get("function"), dict) and t["function"].get("name") == "run_shell")
    ]
    console.print(f"[bold]Tools available: {len(all_tools)}[/bold]\n")

    model = _pick_model()
    system_prompt: str | None = args.system or None
    history: list[dict] = []

    def _build_system() -> str | None:
        """Merge user system prompt with auto-injected memory block."""
        mem_block = memory.context_block()
        parts = [p for p in [mem_block, system_prompt] if p]
        return "\n\n".join(parts) if parts else None

    def _status() -> str:
        turns = len([m for m in history if m["role"] == "user"])
        sys_tag = " [dim]+ system[/dim]" if system_prompt else ""
        mem_tag = f" [dim]· {len(memory)}mem[/dim]" if len(memory) else ""
        return (
            f"[bold yellow]{model}[/bold yellow]{sys_tag}{mem_tag} "
            f"[dim]· {len(all_tools)} tools · {turns} turn{'s' if turns != 1 else ''}[/dim]"
        )

    console.print(f"Using: {_status()}\n")

    async def call_tool(name: str, tool_args: dict) -> str:
        if name in memory_handlers:
            return memory_handlers[name](tool_args)
        if mcp.is_mcp_tool(name):
            return await mcp.call_tool(name, tool_args)
        if name in BUILTIN_HANDLERS:
            return BUILTIN_HANDLERS[name](tool_args)
        return f"Unknown tool: {name}"

    def on_tool_call(name: str, tool_args: dict) -> None:
        color = "magenta" if name in memory_handlers else "yellow"
        console.print(f"\n[{color}]⚙ tool:[/{color}] [bold]{name}[/bold]  [dim]{tool_args}[/dim]")

    AGENT_HELP = """\
[bold]Commands:[/bold]
  [cyan]/model[/cyan]          Change model
  [cyan]/system [msg][/cyan]   Set system prompt
  [cyan]/tools[/cyan]          List available tools
  [cyan]/memory[/cyan]         Show all stored memories
  [cyan]/forget <key>[/cyan]   Delete a memory entry
  [cyan]/new[/cyan]            Clear conversation history
  [cyan]/help[/cyan]           Show this help
  [cyan]/exit[/cyan]           Quit
"""

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            rest = user_input[len(cmd):].strip()

            if cmd == "/exit":
                console.print("[dim]Bye.[/dim]")
                break
            elif cmd == "/new":
                history.clear()
                console.print("[dim]Conversation cleared.[/dim]")
            elif cmd == "/model":
                model = _pick_model()
                console.print(f"\nUsing: {_status()}\n")
            elif cmd == "/system":
                system_prompt = rest or None
                label = f"[green]{system_prompt}[/green]" if system_prompt else "[dim]cleared[/dim]"
                console.print(f"System prompt: {label}")
            elif cmd == "/tools":
                for t in all_tools:
                    fn = t["function"]
                    console.print(f"  [cyan]{fn['name']}[/cyan] — {fn.get('description', '')}")
            elif cmd == "/memory":
                console.print(memory.list_all())
            elif cmd == "/forget":
                if rest:
                    console.print(memory.delete(rest))
                else:
                    console.print("[red]Usage: /forget <key>[/red]")
            elif cmd == "/help":
                rprint(AGENT_HELP)
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
            continue

        effective_system = _build_system()
        messages: list[dict] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        try:
            reply = await agent_loop.run(
                client=_client(),
                model=model,
                messages=messages,
                tools=all_tools,
                call_tool=call_tool,
                console=console,
                on_tool_call=on_tool_call,
            )
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
        except httpx.HTTPStatusError as e:
            console.print(f"[red]HTTP {e.response.status_code}:[/red] {e.response.text}")
        except Exception as e:
            console.print(f"[red]{type(e).__name__}:[/red] {e!r}")
            continue

        console.print(f"\n[dim]{_status()}\n")

    await mcp.stop()


# ---------------------------------------------------------------------------
# Non-interactive commands
# ---------------------------------------------------------------------------

def cmd_providers(_args: argparse.Namespace) -> None:
    data = _raw_get("/providers")
    for p in data.get("data", []):
        print(p["id"])


def cmd_models(args: argparse.Namespace) -> None:
    path = f"/models/{args.provider}" if args.provider else "/models"
    data = _raw_get(path)
    models = data.get("data", [])
    if not models:
        print("No models found.")
        return
    col_w = max(len(m["id"]) for m in models) + 2
    for m in models:
        name = m.get("name") or ""
        ctx = m.get("context_length")
        ctx_str = f"ctx:{ctx:,}" if ctx else ""
        free = "[free]" if m.get("is_free") else ""
        print(f"{m['id']:<{col_w}} {name:<40} {ctx_str:<15} {free}")


def cmd_chat(args: argparse.Namespace) -> None:
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.message})

    kwargs: dict = {"model": args.model, "messages": messages}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens

    response = _client().chat.completions.create(**kwargs)
    print(response.choices[0].message.content)

    usage = response.usage
    if usage:
        print(
            f"\n[tokens: prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"total={usage.total_tokens}]",
            file=sys.stderr,
        )


def cmd_stream(args: argparse.Namespace) -> None:
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.message})

    kwargs: dict = {"model": args.model, "messages": messages, "stream": True}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens

    stream = _client().chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="client.py",
        description="LLM Proxy CLI — OpenAI-SDK-based client",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ui", help="Interactive chat UI (default when no command given)")

    p_agent = sub.add_parser("agent", help="Interactive agent UI with tool use + MCP")
    p_agent.add_argument("--mcp", metavar="FILE", default="mcp.json", help="MCP config file (default: mcp.json)")
    p_agent.add_argument("--no-builtin", action="store_true", help="Disable built-in tools")
    p_agent.add_argument("--system", metavar="MSG", help="System prompt")

    sub.add_parser("providers", help="List active providers")

    p_models = sub.add_parser("models", help="List available models")
    p_models.add_argument("--provider", metavar="NAME", help="Filter by provider")

    def add_chat_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("model", help="Model in provider/model format")
        p.add_argument("message", help="User message")
        p.add_argument("--system", metavar="MSG", help="System prompt")
        p.add_argument("--temperature", type=float, metavar="F")
        p.add_argument("--max-tokens", type=int, metavar="N", dest="max_tokens")

    add_chat_args(sub.add_parser("chat", help="Non-streaming chat completion"))
    add_chat_args(sub.add_parser("stream", help="Streaming chat completion"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "ui": cmd_ui,
        "agent": cmd_agent,
        "providers": cmd_providers,
        "models": cmd_models,
        "chat": cmd_chat,
        "stream": cmd_stream,
    }

    try:
        commands.get(args.command or "ui", cmd_ui)(args)
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
