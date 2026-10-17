"""Shared rate limits (app/limits.py): windows, all-or-none slots, provider cooldowns, and how the
routes behave over a limit (spoken "busy", memories still saved, 429 with Retry-After)."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from app.ai import BUSY, AIServices, limited
from app.config import Settings
from app.limits import Limiter, MemorySlots, RateLimited, Rule, per, provider_busy
from tests.test_api import CAREGIVER, DEVICE, client, log  # noqa: F401  (client is a fixture)

UTC = timezone.utc


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- windows and slots

def test_windows():
    t = datetime(2026, 10, 17, 15, 42, 7, tzinfo=UTC)
    assert Rule("b", 1, "minute").span(t) == (datetime(2026, 10, 17, 15, 42, tzinfo=UTC), datetime(2026, 10, 17, 15, 43, tzinfo=UTC))
    assert Rule("b", 1, "hour").span(t)[0] == datetime(2026, 10, 17, 15, tzinfo=UTC)
    # Gemini's daily quota resets at midnight Pacific: 07:00 UTC in October (PDT)
    assert Rule("b", 1, "day").span(t) == (datetime(2026, 10, 17, 7, tzinfo=UTC), datetime(2026, 10, 18, 7, tzinfo=UTC))
    assert Rule("b", 1, "day").span(datetime(2026, 10, 17, 6, 59, tzinfo=UTC))[0] == datetime(2026, 10, 16, 7, tzinfo=UTC)
    # across the November DST change the day is 25 hours long, and still ends at local midnight
    start, end = Rule("b", 1, "day").span(datetime(2026, 11, 1, 12, tzinfo=UTC))
    assert (start, end) == (datetime(2026, 11, 1, 7, tzinfo=UTC), datetime(2026, 11, 2, 8, tzinfo=UTC))
    assert Rule("b", 1, "month").span(t) == (datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))


def test_per_skips_missing_caps():
    assert per("b") == [] and [r.window for r in per("b", 5, 0)] == ["minute"]
    assert [(r.limit, r.window) for r in per("b", 5, 50)] == [(5, "minute"), (50, "day")]


def test_slots_are_all_or_none():
    now = [datetime(2026, 10, 17, 12, tzinfo=UTC)]
    store = MemorySlots()
    lim = Limiter(store, clock=lambda: now[0])
    rules = [Rule("gemini:m", 3, "minute"), Rule("gemini:m", 2, "day")]
    run(lim.take(rules))
    run(lim.take(rules))
    with pytest.raises(RateLimited) as full:
        run(lim.take(rules))  # the day is full: the minute slot isn't spent either
    assert full.value.bucket == "gemini:m" and "day" in full.value.why
    assert full.value.retry_after == (datetime(2026, 10, 18, 7, tzinfo=UTC) - now[0]).total_seconds()
    minute_key = ("gemini:m:minute", datetime(2026, 10, 17, 12, tzinfo=UTC))
    assert store._rate_slots[minute_key] == 2
    run(lim.take([Rule("other", 1, "minute")]))  # other buckets are separate


def test_cooldown_blocks_every_rule_of_that_bucket():
    now = datetime(2026, 10, 17, 12, tzinfo=UTC)
    lim = Limiter(MemorySlots(), clock=lambda: now)
    run(lim.cool_down("gemini:m", 40))
    with pytest.raises(RateLimited) as busy:
        run(lim.take(per("gemini:m", 100, 1000)))
    assert busy.value.why == "cooling down" and busy.value.retry_after == 40
    run(lim.take(per("gemini:tts", 100)))


def test_provider_busy_reads_googles_retry_delay():
    google = SimpleNamespace(code=429, details={"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]}})
    assert provider_busy(google) == 38
    assert provider_busy(SimpleNamespace(code=503, details={})) == 60  # overloaded, no delay given
    assert provider_busy(SimpleNamespace(status_code=429, response=SimpleNamespace(headers={"retry-after": "5"}))) == 6
    assert provider_busy(SimpleNamespace(code=400)) is None and provider_busy(ValueError("x")) is None


# ---------------------------------------------------------------- model calls

class Flaky(AIServices):
    """AIServices with one guarded call standing in for a provider."""

    def __init__(self, settings, limiter, fail=None):
        super().__init__(settings, limiter)
        self.fail, self.calls = fail, 0

    @limited("text")
    async def ask(self):
        self.calls += 1
        if self.fail:
            raise self.fail
        return "ok"


def settings(**kw):
    return Settings(_env_file=None, gemini_rpm=2, gemini_rpd=100, **kw)


def test_model_calls_take_slots_and_stop_at_the_cap():
    ai = Flaky(settings(), Limiter(MemorySlots()))
    assert run(ai.ask()) == "ok" and run(ai.ask()) == "ok"
    with pytest.raises(RateLimited):
        run(ai.ask())
    assert ai.calls == 2  # the third never reached the provider


def test_a_provider_429_cools_the_model_down_for_everyone():
    store = MemorySlots()
    busy = SimpleNamespace(code=429, details={"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "20s"}]}})
    first = Flaky(settings(), Limiter(store), fail=type("ClientError", (Exception,), {"code": 429, "details": busy.details})())
    with pytest.raises(RateLimited) as limited_:
        run(first.ask())
    assert limited_.value.why == "provider busy" and limited_.value.retry_after == 21
    other_instance = Flaky(settings(), Limiter(store))  # another server, same database
    with pytest.raises(RateLimited):
        run(other_instance.ask())
    assert other_instance.calls == 0


def test_other_errors_pass_through_untouched():
    ai = Flaky(settings(), Limiter(MemorySlots()), fail=ValueError("bad request"))
    with pytest.raises(ValueError):
        run(ai.ask())


def test_rules_follow_the_provider_and_model():
    ai = AIServices(settings(vision_provider="none", tts_provider="openai", openai_rpm=7, openai_rpd=70))
    assert {r.bucket for r in ai.rules("text")} == {"gemini:gemini-3.8-flash"}
    assert {r.bucket for r in ai.rules("embed")} == {"gemini:gemini-embedding-2"}
    assert [(r.bucket, r.limit) for r in ai.rules("tts")] == [("openai", 7), ("openai", 70)]
    assert ai.rules("vision") == []  # VISION_PROVIDER=none makes no call
    assert {r.bucket for r in AIServices(settings(llm_provider="claude")).rules("text")} == {"claude"}


# ---------------------------------------------------------------- routes

def over_limit(*a, **k):
    raise RateLimited("gemini:gemini-embedding-2", 30)


def test_a_question_over_the_limit_is_answered_busy(client):  # noqa: F811
    client.app_state.ai.embed = over_limit
    resp = client.post("/api/v1/memory/query", headers=DEVICE, files={"audio": ("q.wav", b"where are my keys", "audio/wav")})
    assert resp.status_code == 200 and resp.headers["x-answered-by"] == "busy"
    assert unquote(resp.headers["x-answer"]) == BUSY


def test_a_memory_is_still_saved_when_indexing_is_limited(client):  # noqa: F811
    client.app_state.ai.embed_many = over_limit
    resp = log(client, "I am putting my keys on the counter")
    assert resp.status_code == 200 and len(client.app_state.db.rows) == 1


def test_each_device_has_a_request_cap(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(client.settings, "device_rpm", 2)
    codes = [log(client, "keys on the hook").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    resp = log(client, "keys on the hook")
    assert int(resp.headers["retry-after"]) >= 1 and resp.json()["retry_after"] >= 1


def test_each_caregiver_has_a_cap_on_ai_dashboard_calls(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(client.settings, "caregiver_ai_rpm", 1)
    log(client, "I am putting my keys on the counter")
    assert client.get("/api/v1/dashboard/search?q=keys", headers=CAREGIVER).status_code == 200
    assert client.get("/api/v1/dashboard/search?q=keys", headers=CAREGIVER).status_code == 429
