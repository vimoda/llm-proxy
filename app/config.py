from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    nvidia_api_key: str = ""

    # Rate limiting (in-memory, per IP)
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 60   # requests per minute
    rate_limit_tpm: int = 100_000  # tokens per minute (0 = disabled)

    # Auth — comma-separated API keys string.
    # If empty, auth is disabled (useful for local dev).
    # .env example:  API_KEYS=key-one,key-two
    api_keys: str = ""

    def get_api_keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
