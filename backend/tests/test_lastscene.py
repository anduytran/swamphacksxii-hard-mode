"""Where the wearer is, compared with an item's LastScene (app/lastscene.py): the sentence and when it's said."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import unquote

from app.ai import Command, PhotoCheck
from app.config import Settings
from app.geo import Location
from app.lastscene import ItemPosition, compare, lastscene_note, spoken_distance
from tests.test_api import DEVICE, client  # noqa: F401  (client is a fixture)

REITZ = (29.6465, -82.3475)
HOME = (29.6400, -82.3600)  # about 1.4 km from the Reitz Union
NOW = datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)
SETTINGS = Settings(_env_file=None)


def at(spot, place=None, accuracy=20.0):
    return ItemPosition(spot[0], spot[1], accuracy, place, NOW)


def here(spot, accuracy=20.0):
    return Location(spot[0], spot[1], accuracy, "wifi")


def test_the_sentence_says_here_or_how_far_and_where():
    labels = {"p-reitz": "Reitz Union", "p-home": "Home"}
    assert compare(at(REITZ, "p-reitz"), here(REITZ), "p-reitz", labels) == "Its LastScene is here at Reitz Union."
    assert compare(at(REITZ), here((29.6466, -82.3476)), None, {}) == "Its LastScene is right around here."
    far = compare(at(HOME, "p-home"), here(REITZ), "p-reitz", labels)
    assert far.startswith("Its LastScene is at Home, about 1.") and far.endswith(" kilometers from you.")
    assert compare(at((29.6475, -82.3475)), here(REITZ), None, {}) == "Its LastScene is about 100 meters from you."
    # Rough fixes: within their accuracy it counts as here.
    assert "here" in compare(at((29.6475, -82.3475), accuracy=80), here(REITZ, accuracy=60), None, {})


def test_spoken_distances_are_rounded():
    assert spoken_distance(4) == "10 meters"
    assert spoken_distance(87) == "90 meters"
    assert spoken_distance(312) == "300 meters"
    assert spoken_distance(1349) == "1.3 kilometers"


class FakeGeo:
    async def locate(self, aps):
        return Location(*REITZ, 15.0, "wifi") if aps else None


class FakeDB:
    def __init__(self, item_at=None, recent=None, places=()):
        self.item_at, self.recent, self.places = item_at, recent, list(places)

    async def household_of_device(self, device_id):
        return ("house-1", False)

    async def household_device_ids(self, household_id):
        return ["cam@clients"]

    async def lastscene_position(self, scope, device_ids, item):
        assert (scope, device_ids, item) == ("house-1", ["cam@clients"], "pin")
        return self.item_at

    async def recent_position(self, device_ids, minutes):
        return self.recent

    async def places_for(self, household_id):
        return self.places


def place(pid, name, spot):
    return SimpleNamespace(id=pid, name=name, kind="private", lat=spot[0], lng=spot[1], radius_m=75, visits=3)


def note(db, **where):
    return asyncio.run(lastscene_note(db, FakeGeo(), SETTINGS, "cam@clients", "Pin", **where))


def test_the_wearers_position_comes_from_this_request_else_their_last_known_one():
    places = [place("p-reitz", "Reitz Union", REITZ), place("p-home", "Home", HOME)]
    item = (*HOME, 20.0, "p-home", NOW)
    assert note(FakeDB(item, places=places), wifi='[{"bssid": "aa:bb:cc:dd:ee:ff", "signal": -50}, {"bssid": "11:22:33:44:55:66", "signal": -60}]').startswith("Its LastScene is at Home")
    assert note(FakeDB(item, recent=(*REITZ, 30.0), places=places)).startswith("Its LastScene is at Home")
    assert note(FakeDB((*REITZ, 20.0, "p-reitz", NOW), recent=(*REITZ, 30.0), places=places)) == \
        "Its LastScene is here at Reitz Union."
    assert note(FakeDB(item, recent=None, places=places)) is None  # where the wearer is isn't known
    assert note(FakeDB(None, recent=(*REITZ, 30.0), places=places)) is None  # the item was never located


def test_an_ask_adds_where_its_lastscene_is(client):  # noqa: F811
    """Through POST /api/v1/memory/voice: the answer, then the LastScene sentence."""
    db = client.app_state.db
    fake = FakeDB((*HOME, 20.0, "p-home", NOW), recent=(*REITZ, 30.0),
                  places=[place("p-reitz", "Reitz Union", REITZ), place("p-home", "Home", HOME)])
    for name in ("lastscene_position", "recent_position", "places_for", "household_of_device", "household_device_ids"):
        setattr(db, name, getattr(fake, name))

    async def decipher(words):
        return Command(intent="ask", item="pin")

    async def look(item, words, photos):
        return PhotoCheck(visible=False)

    client.app_state.ai.decipher, client.app_state.ai.look_for = decipher, look
    files = [("audio", ("v.wav", b"Last Scene, where is my pin", "audio/wav")), ("image", ("p.jpg", b"\xff\xd8x", "image/jpeg"))]
    resp = client.post("/api/v1/memory/voice", headers=DEVICE, files=files, data={"wake": "true"})
    assert resp.status_code == 200
    answer = unquote(resp.headers["x-answer"])
    assert "Its LastScene is at Home, about 1." in answer and answer.endswith("kilometers from you.")
