from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """FastAPI dependency — validates Bearer token against configured api_keys.

    If api_keys is empty (default), auth is disabled so local dev works out of the box.
    """
    keys = settings.get_api_keys()
    if not keys:
        return

    if credentials is None or credentials.credentials not in keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )
