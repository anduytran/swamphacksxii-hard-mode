"""MCP keys and the MCP rate counter against the real database.

Opt-in (writes to DATABASE_URL, cleans up after itself):
    RUN_DB_TESTS=1 .venv/bin/python -m pytest tests/test_mcp_integration.py
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.auth import Principal, require_caregiver
from app.db_mcp import mcp_key_hash
from app.main import create_app

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_DB_TESTS"), reason="set RUN_DB_TESTS=1 to run against DATABASE_URL")
RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def setup():
    app = create_app(use_lifespan=False)
    owner = f"it-{RUN}-owner"
    app.dependency_overrides[require_caregiver] = lambda: Principal(sub=owner, claims={})
    with TestClient(app) as client:
        yield client, app.state.db
        run = client.portal.call
        run(app.state.db.pool.execute, "DELETE FROM mcp_rate WHERE subject LIKE $1", f"it-{RUN}-%")
        run(app.state.db.pool.execute, "DELETE FROM households WHERE created_by LIKE $1", f"it-{RUN}-%")  # keys cascade


def test_key_lifecycle_and_rate_counter(setup):
    c, db = setup
    run = c.portal.call
    hh = c.post("/api/v1/households", json={"name": f"MCP ({RUN})"}).json()["id"]
    made = c.post(f"/api/v1/households/{hh}/mcp-keys", json={"label": "script"}).json()
    assert run(db.mcp_key_caller, mcp_key_hash(made["key"])) == (made["id"], hh, f"it-{RUN}-owner")
    assert c.get(f"/api/v1/households/{hh}/mcp-keys").json()["keys"][0]["last_used_at"]
    assert c.delete(f"/api/v1/households/{hh}/mcp-keys/{made['id']}").status_code == 200
    assert run(db.mcp_key_caller, mcp_key_hash(made["key"])) is None

    subject = f"it-{RUN}-rate"
    assert [run(db.mcp_rate_hit, subject, 60) for _ in range(3)] == [1, 2, 3]
    run(db.pool.execute, "UPDATE mcp_rate SET window_start = now() - interval '2 minutes' WHERE subject = $1", subject)
    assert run(db.mcp_rate_hit, subject, 60) == 1  # a new window
