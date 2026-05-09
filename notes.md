RATE LIMIT Implementado. Resumen:

  ---
  Dos niveles de rate limiting

  RPM (requests per minute) — via slowapi por IP:
  - @limiter.limit(rpm_limit) en POST /v1/chat/completions
  - Retorna 429 con header Retry-After en formato OpenAI-compatible
  - Configurable: RATE_LIMIT_RPM=60

  TPM (tokens per minute) — sliding window en memoria por IP:
  - Pre-check: estima tokens de entrada (len(content) // 4) antes de llamar al provider
  - Post-charge: registra tokens reales de salida (usage.completion_tokens) después de la respuesta
  - Configurable: RATE_LIMIT_TPM=100000 (0 = desactivado)
  - Ventana deslizante de 60s con lock thread-safe

  # app/rate_limiter.py — sliding window simplificada
  _tpm_window: dict[str, list[tuple[float, int]]] = defaultdict(list)

  Variables de entorno:
  RATE_LIMIT_ENABLED=true
  RATE_LIMIT_RPM=60
  RATE_LIMIT_TPM=100000

  ▎ Para deployments multi-proceso o distribuidos, reemplazar el store in-memory con Redis: slowapi soporta
  Limiter(storage_uri="redis://...") y el TPM store puede usar redis-py.