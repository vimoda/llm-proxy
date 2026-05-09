from typing import Annotated, Any, Literal, Optional, Union
from pydantic import BaseModel, Field


# --- Models list ---

class ModelArchitecture(BaseModel):
    modality: Optional[str] = None
    input_modalities: list[str] = []
    output_modalities: list[str] = []
    tokenizer: Optional[str] = None


class ModelPricing(BaseModel):
    prompt: Optional[str] = None            # price per token (string to avoid float precision)
    completion: Optional[str] = None
    input_cache_read: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    is_free: bool = False
    object: str = "model"
    owned_by: str = ""
    created: Optional[int] = None
    name: Optional[str] = None
    description: Optional[str] = None
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    architecture: Optional[ModelArchitecture] = None
    pricing: Optional[ModelPricing] = None
    supported_parameters: list[str] = []


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ProviderInfo(BaseModel):
    id: str


class ProviderListResponse(BaseModel):
    object: str = "list"
    data: list[ProviderInfo]


# --- Tool calling ---

class ToolFunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionDefinition


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


# --- Multimodal content parts ---

class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


class ImageUrlContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[
    TextContentPart | ImageUrlContentPart,
    Field(discriminator="type"),
]


# --- Request / Response ---

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str | list[ContentPart]] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # only when role="tool"


class ChatCompletionRequest(BaseModel):
    # Format: "provider/model"  e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Optional[float] = None  # string to avoid float precision issues


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    model: str
    provider: Optional[str] = None
    choices: list[ChatCompletionChoice]
    usage: Usage
