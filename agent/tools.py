"""Built-in agent tools: read_file, list_directory, run_shell."""

import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


def _list_directory(path: str = ".") -> str:
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = []
        for e in entries:
            kind = "FILE" if e.is_file() else "DIR "
            size = e.stat().st_size if e.is_file() else ""
            lines.append(f"{kind}  {e.name}  {size}")
        return "\n".join(lines) or "(empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


def _run_shell(command: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as e:
        return f"Error running command: {e}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current dir)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)."},
                },
                "required": ["command"],
            },
        },
    },
]

BUILTIN_HANDLERS: dict[str, callable] = {
    "read_file": lambda args: _read_file(args["path"]),
    "list_directory": lambda args: _list_directory(args.get("path", ".")),
    "run_shell": lambda args: _run_shell(args["command"], args.get("timeout", 30)),
}
