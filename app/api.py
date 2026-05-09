from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.auth import require_api_key
from app.models import ChatCompletionRequest, ChatCompletionResponse, ModelListResponse, ProviderInfo, ProviderListResponse, TextContentPart
from app.providers import get_provider, list_models, list_provider_names
from app.rate_limiter import rpm_limit, check_tpm, limiter

router = APIRouter(dependencies=[Depends(require_api_key)])

@router.get("/providers", response_model=ProviderListResponse)
async def get_providers() -> ProviderListResponse:
    return ProviderListResponse(data=[ProviderInfo(id=name) for name in list_provider_names()])


# Listar Todos los modelos disponibles en todos los proveedores
@router.get("/models", response_model=ModelListResponse)
async def get_models() -> ModelListResponse:
    return ModelListResponse(data=await list_models())

# Listar los modelos de un proveedor específico
@router.get("/models/{provider_name}", response_model=ModelListResponse)
async def get_provider_models(provider_name: str) -> ModelListResponse:
    try:
        provider = get_provider(provider_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found.")
    models = await provider.list_models()
    return ModelListResponse(data=sorted(models, key=lambda m: m.id.lower()))

# Endpoint para chat completions (sin streaming)
@router.post("/chat/completions", response_model=ChatCompletionResponse)
@limiter.limit(rpm_limit)  # type: ignore[misc]
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
) -> StreamingResponse | ChatCompletionResponse:
    if "/" not in body.model:
        raise HTTPException(
            status_code=400,
            detail="Model must be in 'provider/model' format, e.g. 'openai/gpt-4o'",
        )

    provider_name, model = body.model.split("/", 1)

    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # TPM pre-check: estimate input tokens (1 token ≈ 4 chars)
    ip = request.client.host if request.client else "unknown"

    def _estimate_tokens(msg: ChatCompletionRequest) -> int:
        total = 0
        for m in msg.messages:
            if m.content is None:
                continue
            if isinstance(m.content, str):
                total += len(m.content) // 4
            else:
                total += sum(len(p.text) // 4 for p in m.content if isinstance(p, TextContentPart))
        return total

    estimated_input_tokens = _estimate_tokens(body)
    allowed, current = check_tpm(ip, estimated_input_tokens)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Token rate limit exceeded ({current} tokens used this minute).",
        )

    if body.stream:
        return StreamingResponse(
            provider.stream(body, model),
            media_type="text/event-stream",
        )

    response = await provider.complete(body, model)

    # TPM post-charge: record actual output tokens
    output_tokens = response.usage.completion_tokens
    check_tpm(ip, output_tokens)

    return response
