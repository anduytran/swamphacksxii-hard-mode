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
        if self.item_at is None:
            return None
        lat, lng, acc, place_id, at = self.item_at
        return {"lat": lat, "lng": lng, "accuracy_m": acc, "place_id": place_id, "at": at, "wifi": {}}

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


# ---------------------------------------------------------------- Wi-Fi fingerprints (wifi_places.py)

from app.intent import place_command  # noqa: E402
from app.lastscene import compare_scans  # noqa: E402
from app.wifi_places import best_place, from_wifi_json, merge, similarity  # noqa: E402

REITZ_SCAN = {"aa:00:00:00:00:01": -60, "aa:00:00:00:00:02": -64, "aa:00:00:00:00:03": -70, "aa:00:00:00:00:04": -78}
REITZ_LATER = {"aa:00:00:00:00:01": -63, "aa:00:00:00:00:02": -61, "aa:00:00:00:00:03": -75, "aa:00:00:00:00:09": -80}
HOME_SCAN = {"bb:00:00:00:00:01": -50, "bb:00:00:00:00:02": -66, "bb:00:00:00:00:03": -72}
PLACES = [{"id": "w1", "name": "reitz union", "aps": REITZ_SCAN}, {"id": "w2", "name": "Home", "aps": HOME_SCAN}]


def test_similarity_tells_the_same_place_from_another():
    assert similarity(REITZ_SCAN, REITZ_SCAN) == 1.0
    assert similarity(REITZ_SCAN, REITZ_LATER) > 0.5
    assert similarity(REITZ_SCAN, HOME_SCAN) == 0.0
    assert best_place(PLACES, REITZ_LATER)[0]["id"] == "w1"
    assert best_place(PLACES, {"cc:00:00:00:00:01": -60, "cc:00:00:00:00:02": -60, "cc:00:00:00:00:03": -60}) is None
    assert best_place(PLACES, {"aa:00:00:00:00:01": -60}) is None  # one access point isn't a fingerprint
    merged = merge(REITZ_SCAN, REITZ_LATER)
    assert merged["aa:00:00:00:00:09"] == -80 and merged["aa:00:00:00:00:01"] == round(0.7 * -60 + 0.3 * -63)
    assert from_wifi_json('[{"bssid": "AA:00:00:00:00:01", "signal": -60}]') == {"aa:00:00:00:00:01": -60}


def test_the_scan_sentence():
    assert compare_scans(REITZ_SCAN, REITZ_LATER, PLACES) == "Its LastScene is here at Reitz Union."
    assert compare_scans(HOME_SCAN, REITZ_LATER, PLACES) == "Its LastScene is at Home, not here."
    assert compare_scans(REITZ_SCAN, REITZ_LATER, []) == "Its LastScene is right around here."
    assert compare_scans(HOME_SCAN, REITZ_LATER, []) == "Its LastScene is somewhere else, not where you are now."


def test_naming_a_place_by_voice():
    assert place_command("this is the Reitz Union") == "Reitz Union"
    assert place_command("We're in the kitchen.") == "kitchen"
    assert place_command("I'm at home") == "home"
    assert place_command("remember this place as Marston Library") == "Marston Library"
    for not_a_place in ("this is my pin", "I'm at a loss", "where is my pin", "I'm putting my keys on the desk",
                        "this is it", "I'm in a hurry"):
        assert place_command(not_a_place) is None, not_a_place


class ScanDB(FakeDB):
    """FakeDB with Wi-Fi: the item's scan, the wearer's latest scan, and named places."""
    def __init__(self, item_scan, recent_scan, places=()):
        super().__init__((*REITZ, 20.0, None, NOW), recent=None)
        self.item_scan, self.recent, self.named = item_scan, recent_scan, list(places)

    async def lastscene_position(self, scope, device_ids, item):
        return {"lat": None, "lng": None, "accuracy_m": None, "place_id": None, "at": NOW, "wifi": self.item_scan}

    async def recent_scan(self, device_ids, minutes):
        return self.recent

    async def wifi_places(self, scope):
        return self.named


def test_fingerprints_answer_without_any_position():
    wifi = json_scan(REITZ_LATER)
    assert asyncio.run(lastscene_note(ScanDB(REITZ_SCAN, None, PLACES), FakeGeo(), SETTINGS, "cam@clients", "pin",
                                      wifi=wifi)) == "Its LastScene is here at Reitz Union."
    # No scan with this request: the wearer's latest one.
    assert asyncio.run(lastscene_note(ScanDB(HOME_SCAN, REITZ_LATER, PLACES), FakeGeo(), SETTINGS, "cam@clients",
                                      "pin")) == "Its LastScene is at Home, not here."


def json_scan(scan):
    import json
    return json.dumps([{"bssid": k, "signal": v} for k, v in scan.items()])


def test_this_is_a_place_saves_the_scan_and_says_so(client):  # noqa: F811
    db = client.app_state.db
    saved = {}

    async def wifi_places(scope):
        return []

    async def save_wifi_place(scope, name, aps, place_id=None):
        saved.update(scope=scope, name=name, aps=aps)
        return {"id": "w1", "name": name, "samples": 1}

    db.wifi_places, db.save_wifi_place = wifi_places, save_wifi_place
    files = [("audio", ("v.wav", b"Last Scene, this is the Reitz Union", "audio/wav"))]
    resp = client.post("/api/v1/memory/voice", headers=DEVICE, files=files,
                       data={"wake": "true", "wifi": json_scan(REITZ_SCAN)})
    assert resp.status_code == 200 and resp.headers["x-intent"] == "place"
    assert unquote(resp.headers["x-answer"]) == "Got it, this is Reitz Union."
    assert saved["name"] == "Reitz Union" and saved["aps"] == REITZ_SCAN
    thin = client.post("/api/v1/memory/voice", headers=DEVICE, files=files,
                       data={"wake": "true", "wifi": json_scan({"aa:00:00:00:00:01": -60})})
    assert unquote(thin.headers["x-answer"]) == "I can't hear enough Wi-Fi here to remember this place."
