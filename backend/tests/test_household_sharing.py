"""Sharing inside a household: any member can invite, and everyone sets their own display name."""
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app import auth
from app.config import Settings, get_settings
from app.db_places import Household
from app.main import create_app

DOMAIN, AUDIENCE = "test.auth0.local", "https://api.lastscene"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
HOME = "11111111-1111-1111-1111-111111111111"


def caller(sub):
    now = int(time.time())
    token = jwt.encode({"sub": sub, "aud": AUDIENCE, "iss": f"https://{DOMAIN}/", "iat": now, "exp": now + 600}, KEY, algorithm="RS256")
    return {"Authorization": f"Bearer {token}"}


OWNER, MEMBER, STRANGER = caller("auth0|sree"), caller("auth0|adam"), caller("auth0|someone")


class HouseholdDB:
    """household_members as a dict: (household, user) -> [role, display_name]."""

    def __init__(self):
        self.members = {(HOME, "auth0|sree"): ["owner", "Dylan Tran"], (HOME, "auth0|adam"): ["member", "Adam"],
                        ("other", "auth0|sree"): ["owner", "Dylan Tran"]}
        self.invites = []

    async def member_role(self, household_id, user_sub):
        m = self.members.get((household_id, user_sub))
        return m[0] if m else None

    async def create_invite(self, household_id, user_sub, hours=72):
        self.invites.append((household_id, user_sub))
        return "ABCD-1234-EF56"

    async def set_display_name(self, user_sub, display_name):
        mine = [m for (h, u), m in self.members.items() if u == user_sub]
        for m in mine:
            m[1] = display_name
        return len(mine)

    async def households_for(self, user_sub):
        return [Household(h, h, False, m[0], "en") for (h, u), m in self.members.items() if u == user_sub]

    async def household_detail(self, household_id):
        return {"members": [{"user_sub": u, "role": m[0], "display_name": m[1]} for (h, u), m in self.members.items() if h == household_id],
                "devices": []}

    async def places_for(self, household_id):
        return []


@pytest.fixture
def client(monkeypatch):
    settings = Settings(auth0_domain=DOMAIN, auth0_audience=AUDIENCE, image_url_secret="test-secret", _env_file=None)
    signing = SimpleNamespace(key=KEY.public_key())
    monkeypatch.setattr(auth, "_jwks_client", lambda domain: SimpleNamespace(get_signing_key_from_jwt=lambda t: signing))
    app = create_app(use_lifespan=False)
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db = HouseholdDB()
    with TestClient(app) as c:
        c.db = app.state.db
        yield c


def test_any_member_can_invite_but_outsiders_cannot(client):
    for who in (OWNER, MEMBER):
        r = client.post(f"/api/v1/households/{HOME}/invites", headers=who)
        assert r.status_code == 200 and r.json()["code"] == "ABCD-1234-EF56"
    assert client.db.invites == [(HOME, "auth0|sree"), (HOME, "auth0|adam")]
    assert client.post(f"/api/v1/households/{HOME}/invites", headers=STRANGER).status_code == 404


def test_members_still_cannot_change_owner_settings(client):
    assert client.patch(f"/api/v1/households/{HOME}", headers=MEMBER, json={"share_public": True}).status_code == 403
    r = client.post(f"/api/v1/households/{HOME}/devices", headers=MEMBER, json={"client_id": "abcd", "pairing_code": "ABCD"})
    assert r.status_code == 403


def test_you_rename_yourself_everywhere_and_only_yourself(client):
    r = client.patch("/api/v1/me", headers=OWNER, json={"display_name": "  Sreeraaghav   Raja "})
    assert r.status_code == 200 and r.json() == {"display_name": "Sreeraaghav Raja"}
    assert client.db.members[(HOME, "auth0|sree")][1] == client.db.members[("other", "auth0|sree")][1] == "Sreeraaghav Raja"
    assert client.db.members[(HOME, "auth0|adam")][1] == "Adam"
    people = client.get("/api/v1/households/me", headers=MEMBER).json()["households"][0]["members"]
    assert {(p["display_name"], p["is_you"]) for p in people} == {("Sreeraaghav Raja", False), ("Adam", True)}


def test_blank_or_long_names_and_no_household(client):
    assert client.patch("/api/v1/me", headers=OWNER, json={"display_name": "   "}).status_code == 422
    assert client.patch("/api/v1/me", headers=OWNER, json={"display_name": "x" * 81}).status_code == 422
    assert client.patch("/api/v1/me", headers=STRANGER, json={"display_name": "Someone"}).status_code == 404
