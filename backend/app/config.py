from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth0
    auth0_domain: str = ""  # e.g. memory-anchor.us.auth0.com
    auth0_audience: str = ""  # API identifier, e.g. https://api.memory-anchor
    # Auth0 SPA application used by the caregiver dashboard (public values).
    auth0_spa_client_id: str = ""
    # Scope the wearable's M2M token must carry (grant it to the M2M app in Auth0).
    device_scope: str = "write:memories"
    # Optional RBAC permission caregivers must carry. Empty = any valid user token.
    caregiver_permission: str = ""

    # TigerData (Postgres + pgvector)
    database_url: str = ""  # postgres://user:pass@host:port/tsdb?sslmode=require

    # Vultr Object Storage (S3-compatible)
    s3_endpoint_url: str = ""  # e.g. https://ewr1.vultrobjects.com
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "memory-anchor"
    presigned_url_ttl_seconds: int = 900

    # AI providers
    openai_api_key: str = ""
    stt_model: str = "whisper-1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_format: str = "wav"  # wav is easiest for the ESP32 I2S speaker to play

    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"

    # Data lifecycle
    retention_hours: int = 72
    query_lookback_hours: int = 72
    dashboard_history_hours: int = 24


@lru_cache
def get_settings() -> Settings:
    return Settings()
