#!/usr/bin/env bash
# Test the raia frame context end to end. From anywhere:
#
#   backend/scripts/test_raia.sh path/to/photo.jpg [label]          offline tests, then one live raia call
#   backend/scripts/test_raia.sh --db path/to/photo.jpg [label]     also the Postgres test (writes to DATABASE_URL,
#                                                                   cleans up after itself; creates raia_contexts)
#
# 1. tests/test_raia.py: the trigger rules (observations and wake photo checks at 0.5 or more, logged items only,
#    30 min cooldown), the JSON context and the "LastScene: " memory, against a mock raia.
# 2. With --db: tests/test_observations_integration.py's raia test, through POST /api/v1/observations.
# 3. scripts/raia_context_test.py: the photo to the real agent with RAIA_AGENT_KEY from backend/.env.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

DB=0
if [ "${1:-}" = "--db" ]; then DB=1; shift; fi
IMAGE="${1:-}"
LABEL="${2:-pin}"
if [ -z "$IMAGE" ]; then
  echo "usage: $0 [--db] path/to/photo.jpg [label]" >&2
  exit 2
fi
case "$IMAGE" in /*) ;; *) IMAGE="$OLDPWD/$IMAGE" ;; esac

echo "== 1. Offline tests (mock raia)"
"$PY" -m pytest -q tests/test_raia.py

if [ "$DB" = 1 ]; then
  echo
  echo "== 2. Postgres test (DATABASE_URL from backend/.env)"
  RUN_DB_TESTS=1 "$PY" -m pytest -q tests/test_observations_integration.py -k raia
fi

echo
echo "== 3. Live raia call"
"$PY" scripts/raia_context_test.py --image "$IMAGE" --label "$LABEL"
