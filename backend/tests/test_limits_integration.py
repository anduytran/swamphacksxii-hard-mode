"""The shared rate-limit counters against the real database: all-or-none, cooldowns, and no
overshoot when many requests race for the last slots.

Opt-in (writes to DATABASE_URL, cleans up after itself):
    RUN_DB_TESTS=1 .venv/bin/python -m pytest tests/test_limits_integration.py
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.limits import Limiter, RateLimited, Rule
from app.main import create_app

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_DB_TESTS"), reason="set RUN_DB_TESTS=1 to run against DATABASE_URL")
RUN = f"it-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def db():
    app = create_app(use_lifespan=False)
    with TestClient(app) as client:
        client.portal.call(app.state.db.ensure_ready)
        yield client.portal.call, app.state.db
        for table in ("rate_slots", "rate_cooldowns"):
            client.portal.call(app.state.db.pool.execute, f"DELETE FROM {table} WHERE bucket LIKE $1", f"{RUN}%")


def test_all_or_none_and_cooldowns(db):
    run, store = db
    lim = Limiter(store)
    rules = [Rule(f"{RUN}:m", 5, "minute"), Rule(f"{RUN}:m", 1, "day")]
    run(lim.take, rules)
    with pytest.raises(RateLimited):
        run(lim.take, rules)  # the day is full, and the minute slot it tried was rolled back
    used = run(store.pool.fetchval, "SELECT used FROM rate_slots WHERE bucket = $1", f"{RUN}:m:minute")
    assert used == 1

    run(lim.cool_down, f"{RUN}:cool", 30)
    run(lim.cool_down, f"{RUN}:cool", 5)  # a shorter cooldown never shortens a longer one
    with pytest.raises(RateLimited) as busy:
        run(lim.take, [Rule(f"{RUN}:cool", 100, "minute")])
    assert 25 <= busy.value.retry_after <= 31


def test_racing_requests_never_overshoot(db):
    run, store = db
    lim = Limiter(store)

    async def race():
        async def one():
            try:
                await lim.take([Rule(f"{RUN}:race", 4, "minute")])
                return True
            except RateLimited:
                return False
        return await asyncio.gather(*(one() for _ in range(12)))

    assert sum(run(race)) == 4


def test_old_windows_are_pruned(db):
    run, store = db
    old = datetime.now(timezone.utc) - timedelta(days=3)
    run(store.pool.execute, "INSERT INTO rate_slots (bucket, window_start, used) VALUES ($1, $2, 1)", f"{RUN}:old", old)
    run(store.prune_limits, datetime.now(timezone.utc) - timedelta(days=2))
    assert run(store.pool.fetchval, "SELECT count(*) FROM rate_slots WHERE bucket = $1", f"{RUN}:old") == 0
