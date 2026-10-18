from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # backend/.env wherever the app is started from; real environment variables win (Vercel).
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")

    # Auth0
    auth0_domain: str = ""  # e.g. lastscene.us.auth0.com
    auth0_audience: str = ""  # API identifier, e.g. https://api.lastscene
    # Auth0 SPA application used by the dashboard (public values).
    auth0_spa_client_id: str = ""
    # Scope the wearable's M2M token must carry (grant it to the M2M app in Auth0).
    device_scope: str = "write:memories"
    # Optional RBAC permission dashboard users must carry. Empty = any valid user token.
    caregiver_permission: str = ""

    # TigerData (Postgres + pgvector)
    database_url: str = ""  # postgres://user:pass@host:port/tsdb?sslmode=require

    # Signs the short-lived photo links handed to the dashboard. Any long random string.
    image_url_secret: str = ""
    image_url_ttl_seconds: int = 900

    # AI providers. Each stage can run on Gemini (default) or the original provider.
    stt_provider: str = "gemini"  # gemini | openai | elevenlabs (Scribe; falls back to Gemini)
    embedding_provider: str = "gemini"  # gemini | openai  (changing it invalidates stored vectors)
    llm_provider: str = "gemini"  # gemini | claude      (answers, item extraction, summaries)
    tts_provider: str = "gemini"  # gemini | openai | elevenlabs (falls back to Gemini speech)
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

    # ElevenLabs speech for the wearable (TTS_PROVIDER=elevenlabs). Billed in characters: capped per
    # month here, and never past what the plan has left (checked with ElevenLabs, keeping a reserve).
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "SAz9YHcvj6GT2YYXdXww"  # River: relaxed, neutral, calm (premade); any voice id works
    # eleven_multilingual_v2: ElevenLabs' most lifelike voice, 1 credit per character (Flash v2.5: ~75 ms, half
    # that). Answers are under 100 characters, so it stays well inside ELEVENLABS_MONTHLY_CHARS (120,000 at most
    # 120,000 credits, under the Creator plan's 131,000), and the plan's own balance is checked before each call.
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_speed: float = 0.9  # 0.7-1.2; a little slower is easier to follow
    elevenlabs_monthly_chars: int = 120000  # Creator plan: 131,000 a month, no overage
    elevenlabs_reserve_chars: int = 500
    elevenlabs_rpm: int = 20
    # ElevenLabs speech-to-text (STT_PROVIDER=elevenlabs): Scribe v2, billed per hour of audio (separate
    # from the voice's characters). Capped in seconds of audio per month; any failure falls back to Gemini.
    elevenlabs_stt_model: str = "scribe_v2"
    elevenlabs_stt_monthly_seconds: int = 36000  # 10 hours of audio
    elevenlabs_stt_rpm: int = 30
    # Hints for speech-to-text (both providers): "Last Scene" and the household's Teach items and aliases,
    # so names like "swamphacks pins" aren't heard as "Swampfax pins".
    stt_hints: bool = True
    # Each wake-command model call (words, then photos) gets this long; then the rules and saved item names
    # decide, so the wearer never waits on a queued free-tier request (seen taking 24-85 s on Oct 17).
    command_timeout_seconds: float = 4.0
    # Picking the photo that shows the item (up to four photos): longer than reading words, and it decides which
    # photo a memory keeps, so it gets more time before memory falls back to photo 1.
    photo_check_timeout_seconds: float = 8.0
    # A spoken answer: past this the wearer hears what the best memory says instead (Gemini free tier can queue
    # a request for 30 s).
    answer_timeout_seconds: float = 5.0
    # The "it's in view right now" check never holds up an answer: once the answer is ready it gets this long.
    in_view_grace_seconds: float = 0.3
    # Wake commands (the words step and the photo check) have their own models: the fastest with the
    # biggest free limits. Benchmarked Oct 17 on the command task: Gemma 4 26B-A4B ~0.7 s for words and
    # ~1.3 s with 4 photos, 3.5 Flash-Lite about the same, 3.1 Flash-Lite 3.6 s median with 26 s spikes.
    gemini_command_model: str = "gemma-4-26b-a4b-it"
    gemini_command_fallback_models: str = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"

    # Rate limits (limits.py), shared by every server instance through Postgres: per minute
    # (*_RPM) and per day (*_RPD, from midnight Pacific like Gemini's quotas); 0 = no cap.
    # Gemini's quotas are per Google Cloud project; AI Studio shows your project's. Keep these
    # a little under them. Over a cap, questions get a spoken "busy", memories are still saved.
    # Per model, "model=per_minute/per_day". AI Studio's free tier for the lastscene project (Oct 17):
    # 3.1 Flash Lite 15/500, 3 Flash 5/20, 3.8 Flash 5/20, each TTS model 3/10, Embedding 2 100/1000.
    # gemini-2.5-flash isn't listed there, so it gets the same cap as the other small quotas.
    gemini_model_limits: str = (
        "gemini-3.1-flash-lite=14/475,gemini-3-flash-preview=4/18,gemini-2.5-flash=4/18,gemini-3.8-flash=4/18,"
        # Not in AI Studio's list for the project yet: kept under the lower published estimates (Gemma 4
        # reportedly 1,500/day; Flash-Lite estimates run 250-1,500/day). Google's 429 still cools them down.
        "gemma-4-26b-a4b-it=14/1400,gemini-3.5-flash-lite=9/240"
    )
    gemini_rpm: int = 4  # any text model not listed above
    gemini_rpd: int = 18
    gemini_tts_rpm: int = 2  # per TTS model
    gemini_tts_rpd: int = 9
    gemini_embedding_rpm: int = 95
    gemini_embedding_rpd: int = 950
    live_tokens_per_day: int = 100  # Gemini Live sessions, all devices together
    openai_rpm: int = 30  # paid providers, when a stage is switched to them
    openai_rpd: int = 300
    claude_rpm: int = 20
    claude_rpd: int = 200
    device_rpm: int = 20  # per device: spoken requests (/memory/log, /query, /voice, /recall)
    device_rpd: int = 1000
    device_observations_rpm: int = 60  # per device: vision pipeline uploads (it sends at most one per 2 s)
    caregiver_ai_rpm: int = 30  # per user: dashboard requests that call a model (search, summary)

    # Wi-Fi positioning: the wearable sends nearby access points; the server asks Google's
    # Geolocation API (when this key is set) and then BeaconDB where that is.
    google_geolocation_api_key: str = ""
    # BeaconDB: free, no key, same request format as Google (community data, so coverage varies).
    # Used without a Google key, past the Google budget, and when Google has no answer. Empty = off.
    beacondb_url: str = "https://api.beacondb.net/v1/geolocate"
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
    # A passive sighting of an item the camera already saw passively this recently refreshes that memory (newest
    # photo, box and time) instead of adding another timeline entry: one memory, one raia read, per item. 0 = off.
    passive_merge_minutes: float = 30.0
    # A passive sighting sent without its own Wi-Fi scan takes the wearable's last known position from this
    # recently (both Pis are worn together), so every save gets a location. 0 = only the sighting's own.
    location_reuse_minutes: float = 10.0
    # Spoken answers about an item add where its LastScene is compared with the wearer ("here at Reitz Union",
    # "at Home, about 1.3 kilometers from you", lastscene.py). Worked out beside the answer; once the answer is
    # ready it gets at most this long, else the answer goes without it.
    lastscene_note_seconds: float = 1.0
    # The dashboard's "From the wearer now" uses the wearer's latest scan or position from this recently.
    wearer_fix_minutes: float = 30.0

    # raia (api.raia2.com) reads JSON context from the frame when one of the user's logged items (a Teach item,
    # or one named in a spoken memory) is seen at RAIA_MIN_CONFIDENCE or more: by the detector in an observation,
    # or by a wake command's photo check (its own 0-1 confidence). At most once per item every RAIA_COOLDOWN_MINUTES,
    # after the response; saved as a memory "LastScene: <summary>" with the item's place indexed (raia.py).
    # The conversation endpoints take only the agent key; the org key is kept for the org-level endpoints.
    raia_agent_key: str = ""
    raia_org_key: str = ""
    raia_api_url: str = "https://api.raia2.com/external"
    raia_min_confidence: float = 0.5
    raia_cooldown_minutes: float = 30.0
    raia_timeout_seconds: float = 60.0

    # Teach page "Needs a look": observations with detector confidence in [min, max) are
    # asked about.
    review_confidence_min: float = 0.35
    review_confidence_max: float = 0.6

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
