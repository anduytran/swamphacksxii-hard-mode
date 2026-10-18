"""Rate limits every server instance shares, so LastScene stays inside its API quotas and budgets.

Vercel runs many short-lived instances, so counters live in Postgres (db_limits.py), not memory.
Two kinds of limit:

  * Providers (outbound): each Gemini model, and OpenAI / Claude when chosen, gets a per-minute
    and a per-day cap below its quota. Gemini's quotas are per Google Cloud project, and its daily
    ones reset at midnight Pacific time, so the "day" window here does too. Google doesn't publish
    the free tier's numbers (AI Studio shows your project's), so the defaults are conservative
    settings (config.py, *_RPM / *_RPD) to raise once you've checked.
  * Callers (inbound): each device and each dashboard user gets a per-minute and per-day cap on the
    routes that spend those calls, so one stuck device or script can't use up the day's quota.

When Google answers 429 / 503 anyway (its limits move with demand), the provider's bucket cools
down for the delay Google asks for, on every instance, instead of retrying into the wall.
Over a limit, callers get RateLimited: routes turn it into a 429 with Retry-After, the spoken
answer says it's busy, and memory saving keeps the memory and skips the extras.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini's daily quotas reset at midnight Pacific
DEFAULT_COOLDOWN_S = 60


class RateLimited(Exception):
    def __init__(self, bucket: str, retry_after: float, why: str = "limit"):
        super().__init__(f"{bucket}: {why}, retry in {retry_after:.0f} s")
        self.bucket, self.retry_after, self.why = bucket, max(1.0, retry_after), why


@dataclass(frozen=True)
class Rule:
    bucket: str  # e.g. "gemini:gemini-3.8-flash", "device:abc@clients:voice"
    limit: int
    window: str  # minute | hour | day (midnight Pacific) | month

    def span(self, now: datetime) -> tuple[datetime, datetime]:
        """The window this moment falls in: (start, end), both UTC."""
        now = now.astimezone(timezone.utc)
        if self.window == "minute":
            start = now.replace(second=0, microsecond=0)
            return start, start + timedelta(minutes=1)
        if self.window == "hour":
            start = now.replace(minute=0, second=0, microsecond=0)
            return start, start + timedelta(hours=1)
        if self.window == "day":
            local = now.astimezone(PACIFIC).replace(hour=0, minute=0, second=0, microsecond=0)
            nxt = (local + timedelta(days=1, hours=2)).replace(hour=0)  # DST-safe "next midnight"
            return local.astimezone(timezone.utc), nxt.astimezone(timezone.utc)
        if self.window == "month":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = (start + timedelta(days=32)).replace(day=1)
            return start, end
        raise ValueError(f"unknown window {self.window!r}")


def per(bucket: str, per_minute: int | None = None, per_day: int | None = None) -> list[Rule]:
    """The usual pair; a limit of 0 or None means no cap on that window."""
    rules = [Rule(bucket, per_minute, "minute")] if per_minute else []
    return rules + ([Rule(bucket, per_day, "day")] if per_day else [])


class Limiter:
    """Takes slots from shared counters. `store` is the database (db_limits.LimitsMixin)."""

    def __init__(self, store, clock=lambda: datetime.now(timezone.utc)):
        self.store, self.clock = store, clock

    async def take(self, rules: list[Rule], cost: int = 1) -> None:
        """`cost` units (requests, or e.g. characters) from every rule, all or none; RateLimited
        if any would go over or is cooling down."""
        if not rules:
            return
        now = self.clock()
        slots = [(r.bucket + ":" + r.window, r.span(now)[0], r.limit, cost) for r in rules]
        result = await self.store.take_slots(slots, sorted({r.bucket for r in rules}), now)
        if result is None:
            return
        kind, key, until = result
        bucket = key.rsplit(":", 1)[0] if kind == "full" else key
        if kind == "full":
            rule = next(r for r in rules if r.bucket + ":" + r.window == key)
            until = rule.span(now)[1]
        raise RateLimited(bucket, (until - now).total_seconds(), "cooling down" if kind == "cooldown" else f"{kind} for this {key.rsplit(':', 1)[1]}")

    async def cool_down(self, bucket: str, seconds: float) -> None:
        await self.store.set_cooldown(bucket, self.clock() + timedelta(seconds=seconds))
        log.warning("Provider busy: %s cools down for %.0f s", bucket, seconds)


def provider_busy(exc: Exception) -> float | None:
    """Seconds to wait if `exc` is a provider saying "too many requests" / "overloaded", else None.
    Works for google-genai (code 429/503, RetryInfo), openai and anthropic (status_code)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code not in (429, 503, 529):
        return None
    details = getattr(exc, "details", None) or {}
    for d in (details.get("error", details) if isinstance(details, dict) else {}).get("details", []) or []:
        if "RetryInfo" in str(d.get("@type", "")) and (m := re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))):
            return min(float(m.group(1)) + 1, 86400)  # a used-up daily quota asks for hours: honor it
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return min(float(headers.get("retry-after")) + 1, 86400)
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_S


class MemorySlots:
    """The database's take_slots / set_cooldown, in memory: for tests and one-process scripts.
    Mix it into a fake database to get the same limits without Postgres."""

    async def take_slots(self, slots, cooldown_buckets, now):
        used = self.__dict__.setdefault("_rate_slots", {})
        cooling = self.__dict__.setdefault("_rate_cooldowns", {})
        hot = [(cooling[b], b) for b in cooldown_buckets if cooling.get(b) and cooling[b] > now]
        if hot:
            until, bucket = max(hot)
            return "cooldown", bucket, until
        for key, start, limit, cost in slots:
            if used.get((key, start), 0) + cost > limit:
                return "full", key, None
        for key, start, _, cost in slots:
            used[(key, start)] = used.get((key, start), 0) + cost
        return None

    async def set_cooldown(self, bucket, until):
        cooling = self.__dict__.setdefault("_rate_cooldowns", {})
        cooling[bucket] = max(until, cooling.get(bucket, until))
