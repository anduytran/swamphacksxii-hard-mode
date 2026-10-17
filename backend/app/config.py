from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # backend/.env wherever the app is started from; real environment variables win (Vercel).
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")

    # Auth0
    auth0_domain: str = ""  # e.g. lastscene.us.auth0.com
    auth0_audience: str = ""  # API identifier, e.g. https://api.lastscene
    # Auth0 SPA application used by the caregiver dashboard (public values).
    auth0_spa_client_id: str = ""
    # Scope the wearable's M2M token must carry (grant it to the M2M app in Auth0).
    device_scope: str = "write:memories"
    # Optional RBAC permission caregivers must carry. Empty = any valid user token.
    caregiver_permission: str = ""

    # TigerData (Postgres + pgvector)
    database_url: str = ""  # postgres://user:pass@host:port/tsdb?sslmode=require

    # Signs the short-lived photo links handed to the dashboard. Any long random string.
    image_url_secret: str = ""
    image_url_ttl_seconds: int = 900

    # AI providers. Each stage can run on Gemini (default) or the original provider.
    stt_provider: str = "gemini"  # gemini | openai
    embedding_provider: str = "gemini"  # gemini | openai  (changing it invalidates stored vectors)
    llm_provider: str = "gemini"  # gemini | claude      (answers, item extraction, summaries)
    tts_provider: str = "gemini"  # gemini | openai
    vision_provider: str = "gemini"  # gemini | none     (describe each photo)

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    gemini_tts_voice: str = "Kore"
    gemini_embedding_model: str = "gemini-embedding-2"

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

    # Wi-Fi positioning: the wearable sends nearby access points; the server asks Google's
    # Geolocation API where that is. Empty = positions only from the device (or none).
    google_geolocation_api_key: str = ""
    # Spend guard: at most this many Google calls per calendar month (UTC), counted in the
    # database before each call; past it, positions come only from the device. Keep it under
    # Google's free monthly allowance. 0 turns Wi-Fi positioning off.
    geolocation_monthly_limit: int = 5000
    # Answers are cached in the database by access-point set, so a familiar spot costs nothing.
    geolocation_cache_days: int = 30

    # Vision pipeline observations (POST /api/v1/observations)
    min_detection_confidence: float = 0.35  # detections below this are dropped
    maybe_confidence: float = 0.5  # below this, answers and the dashboard say "maybe"
    observation_min_interval_seconds: float = 2.0  # per device; faster uploads get 429
    # How long each item's last-seen frame + box is kept. 72 = same as memories; the
    # integration contract proposes 720 (30 days) once the team agrees.
    last_seen_retention_hours: int = 72

    # Run item extraction + photo description after the device's response (True) or
    # before it (False). Use False on serverless hosts that may freeze after responding.
    enrich_in_background: bool = True
    # Vercel Cron sends "Authorization: Bearer <CRON_SECRET>" to the purge endpoint.
    cron_secret: str = ""

    # Indoor floor plan on the dashboard map: "reitz_union" (the SwampHacks venue) or "home".
    floor_plan: str = "reitz_union"

    # Data lifecycle
    retention_hours: int = 72
    query_lookback_hours: int = 72
    dashboard_history_hours: int = 24


@lru_cache
def get_settings() -> Settings:
    return Settings()
