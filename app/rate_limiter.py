"""
Rate limiting for the LLM proxy.

- RPM  : request-level limit via slowapi (in-memory, per client IP).
- TPM  : token-level limit tracked in-process per IP using a sliding window.

For multi-process / distributed deployments replace the in-memory stores
with Redis (slowapi supports it via `storage_uri="redis://..."` and the
TPM store can use redis-py directly).
"""
import time
from collections import defaultdict
from threading import Lock

from slowapi import Limiter  # type: ignore[import-untyped]
from slowapi.util import get_remote_address  # type: ignore[import-untyped]

from app.config import settings


# ---------------------------------------------------------------------------
# RPM — slowapi handles this
# ---------------------------------------------------------------------------

def rpm_limit() -> str:
    """Dynamic limit string read at request time so env changes take effect."""
    if not settings.rate_limit_enabled:
        return "999999/minute"
    return f"{settings.rate_limit_rpm}/minute"


limiter: Limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# TPM — sliding window per IP (tokens consumed in the last 60 s)
# ---------------------------------------------------------------------------

_tpm_lock = Lock()
# ip → list of (timestamp, tokens) tuples
_tpm_window: dict[str, list[tuple[float, int]]] = defaultdict(list)
_WINDOW = 60.0  # seconds


def check_tpm(ip: str, tokens: int) -> tuple[bool, int]:
    """
    Record *tokens* for *ip* and check whether the TPM budget is exceeded.

    Returns (allowed, current_usage).
    If rate_limit_tpm == 0 the check is skipped and always returns (True, 0).
    """
    limit = settings.rate_limit_tpm
    if not settings.rate_limit_enabled or limit == 0:
        return True, 0

    now = time.monotonic()
    cutoff = now - _WINDOW

    with _tpm_lock:
        window = _tpm_window[ip]
        # Evict expired entries
        _tpm_window[ip] = [(ts, t) for ts, t in window if ts > cutoff]
        current = sum(t for _, t in _tpm_window[ip])

        if current + tokens > limit:
            return False, current

        _tpm_window[ip].append((now, tokens))
        return True, current + tokens
