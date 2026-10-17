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
    # The cheapest models with the biggest free quotas first. When one is at its limit, cooling
    # down, overloaded or down, the call moves on to the next (AIServices). Each model has its own
    # quota at Google and its own caps here (GEMINI_MODEL_LIMITS).
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_fallback_models: str = "gemini-3-flash-preview,gemini-2.5-flash,gemini-3.8-flash"
    gemini_tts_model: str = "gemini-3.8-flash-lite-tts"
    gemini_tts_fallback_models: str = "gemini-3.8-flash-tts,gemini-2.5-flash-preview-tts"
    gemini_tts_voice: str = "Kore"
    gemini_embedding_model: str = "gemini-embedding-2"
    # Live conversation mode: the wearable talks with Gemini Live using a one-use token from
    # POST /api/v1/live/token (the model, persona, tools and GEMINI_TTS_VOICE are locked into it).
    live_enabled: bool = False
    gemini_live_model: str = "gemini-3.8-live"
    live_tokens_per_hour: int = 30  # per device

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

    # Rate limits (limits.py), shared by every server instance through Postgres: per minute
    # (*_RPM) and per day (*_RPD, from midnight Pacific like Gemini's quotas); 0 = no cap.
    # Gemini's quotas are per Google Cloud project; AI Studio shows your project's. Keep these
    # a little under them. Over a cap, questions get a spoken "busy", memories are still saved.
    # Per model, "model=per_minute/per_day". Only gemini-3.8-flash's 20/day is confirmed (Google's own
    # 429 on Oct 17); the others are kept under published free-tier figures until AI Studio confirms.
    gemini_model_limits: str = (
        "gemini-3.1-flash-lite=15/900,gemini-3-flash-preview=10/250,gemini-2.5-flash=10/250,gemini-3.8-flash=10/18"
    )
    gemini_rpm: int = 10  # any text model not listed above
    gemini_rpd: int = 20
    gemini_tts_rpm: int = 10
    gemini_tts_rpd: int = 200
    gemini_embedding_rpm: int = 100
    gemini_embedding_rpd: int = 1000
    live_tokens_per_day: int = 100  # Gemini Live sessions, all devices together
    openai_rpm: int = 30  # paid providers, when a stage is switched to them
    openai_rpd: int = 300
    claude_rpm: int = 20
    claude_rpd: int = 200
    device_rpm: int = 20  # per device: spoken requests (/memory/log, /query, /voice, /recall)
    device_rpd: int = 1000
    device_observations_rpm: int = 60  # per device: vision pipeline uploads (it sends at most one per 2 s)
    caregiver_ai_rpm: int = 30  # per caregiver: dashboard requests that call a model (search, summary)

    # Wi-Fi positioning: the wearable sends nearby access points; the server asks Google's
    # Geolocation API where that is. Empty = positions only from the device (or none).
    google_geolocation_api_key: str = ""
    # Spend guard: at most this many Google calls per calendar month (UTC), counted in the
    # database before each call; past it, positions come only from the device. Keep it under
    # Google's free monthly allowance. 0 turns Wi-Fi positioning off.
    geolocation_monthly_limit: int = 5000
    # Answers are cached in the database by access-point set, so a familiar spot costs nothing.
    geolocation_cache_days: int = 30

    # Name public places ("Starbucks, Reitz Union") with Gemini's Grounding with Google Maps.
    # Off by default: past 5,000 free requests a month it is billed. Only public places are
    # sent, in the background after a memory is located, at most MAPS_DAILY_LIMIT calls a day
    # across the app (50 x 31 stays under the free 5,000), and each answer is cached.
    maps_grounding: bool = False
    maps_daily_limit: int = 50
    maps_cache_days: int = 180

    # Vision pipeline observations (POST /api/v1/observations)
    min_detection_confidence: float = 0.35  # detections below this are dropped
    maybe_confidence: float = 0.5  # below this, answers and the dashboard say "maybe"
    observation_min_interval_seconds: float = 2.0  # per device; faster uploads get 429
    # How long each item's last-seen frame + box is kept. 72 = same as memories; the
    # integration contract proposes 720 (30 days) once the team agrees.
    last_seen_retention_hours: int = 72

    # Teach page "Needs a look": observations with detector confidence in [min, max) are
    # asked about, and Gemini box suggestions are capped per household per hour.
    review_confidence_min: float = 0.35
    review_confidence_max: float = 0.6
    suggest_box_per_hour: int = 30

    # Answer "where are my keys?" from a template when one memory clearly has the item and
    # its place, skipping the language model (the wearer then only waits for speech).
    template_answers: bool = True

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
