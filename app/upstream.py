import json

import httpx


class UpstreamError(Exception):
    def __init__(self, status_code: int, body: str, url: str) -> None:
        self.status_code = status_code
        self.body = body
        self.url = url
        super().__init__(f"Upstream {status_code} from {url}: {body}")


async def check_response(response: httpx.Response) -> None:
    if response.is_error:
        body = await response.aread()
        raise UpstreamError(
            status_code=response.status_code,
            body=body.decode(errors="replace"),
            url=str(response.url),
        )


def error_sse(status_code: int, message: str, body: str = "") -> str:
    payload = json.dumps({
        "error": {
            "message": message,
            "type": "upstream_error",
            "code": status_code,
            "body": body,
        }
    })
    return f"data: {payload}\n\n"
