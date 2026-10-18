# Found-then-reopen bug in item_last_seen (for Dylan)

**Bug.** `upsert_last_seen` sets `found_at = NULL` whenever the incoming `captured_at >=` the stored one. It never compares against `found_at`. So a sighting captured *before* "found it" but uploaded *after* it reopens the item: another device in the household uploading late, or any client that sends out of order. (Adam's bridge already orders its own queue and drops its stale queued sightings on "found it", so this is defense in depth.) `rebuild_last_seen` (after `DELETE /observations/{id}`) also drops `found_at`, because it deletes the row and re-inserts it, and `delete_observation` deletes the row first when the deleted observation is the current last-seen.

**Evidence.** I ran this on real Postgres 16.2 with pgvector 0.6.2 (pgserver's bundled binaries, no Docker), on a copy of `backend/`. Before the fix, both new tests in `tests/test_observations_integration.py` fail:
- `test_late_upload_from_before_found_does_not_reopen`: sighting A at t0, found at t2, then B (captured at t1) arrives, and `found_at` comes back `None`.
- `test_delete_keeps_found_when_remaining_sighting_is_older`: delete an older or the current sighting while the item is found, and `found_at` comes back `None`.

**Fix** (`backend/app/db_observations.py` only; no API, response or schema changes):
- Upsert: the record still moves to the newest sighting, but `found_at = CASE WHEN EXCLUDED.captured_at > found_at THEN NULL ELSE found_at END`. Only a sighting taken after "found it" reopens the item.
- Rebuild: reads `found_at` first, rebuilds as before, then restores it if the remaining latest sighting is `<= found_at`.
- `delete_observation` no longer deletes the row in the device's *current* scope, because rebuild replaces that row. It still deletes a row left in an older, pre-household scope. A third new test covers that case.

**Results after the fix:** DB integration tests 6 passed (3 existing + 3 new). Unit suite 51 passed, 6 skipped. `tests/test_api.py::test_startup_survives_missing_settings...` fails whenever `DATABASE_URL`/`IMAGE_URL_SECRET` are set in the environment, with or without this patch, so run the unit suite without them.

**Apply and run:** from the repo root, `git apply docs/for-dylan/reopen_fix.patch`. Then in `backend/`: `RUN_DB_TESTS=1 DATABASE_URL=postgresql://postgres@127.0.0.1:55432/tsdb IMAGE_URL_SECRET=test python -m pytest tests/test_observations_integration.py`. Use a scratch DB, never production; the tests write to it and clean up afterwards.

**Dashboard (not patched):** `found_at` is not reachable from any dashboard endpoint. `/dashboard/items` is built from `memory_items`/`memory_logs` and never reads `item_last_seen`. Proposed change: add `found_at` per item to `/dashboard/items` with an additive key, using one query: `SELECT item, found_at FROM item_last_seen WHERE scope IN (household ids of the caller's devices)`. Then in the Items page (now the React app in `web/`), show a "Found <timeAgo>" chip, or dim the item, when `found_at` is newer than `last_seen`.
