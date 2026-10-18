"""The vision pipeline's observation format and the text it becomes."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.observations import Observation, summary, text_chunks, too_far_ahead

SETTINGS = Settings(_env_file=None)


def payload(**over):
    base = {
        "observation_id": str(uuid.uuid4()),
        "captured_at": "2026-10-17T14:03:12.481Z",
        "source": "auto",
        "pipeline": {"name": "lastscene-vision", "version": "0.3.0", "detector": "yolo-world", "matcher": "orb"},
        "image_size": {"width": 1280, "height": 720},
        "scene": {"id": "kitchen-01", "label": "kitchen counter", "match_score": 0.82},
        "objects": [{"label": "Keys", "confidence": 0.71, "bbox": [0.42, 0.55, 0.08, 0.06], "track_id": "keys-3", "event": "placed"}],
    }
    base.update(over)
    return base


def test_contract_example_is_valid_and_normalised():
    obs = Observation.model_validate(payload())
    assert obs.objects[0].label == "keys"
    assert obs.captured_at.tzinfo is not None and obs.captured_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "over, problem",
    [
        ({"source": "manual"}, "source"),
        ({"captured_at": "2026-10-17T14:03:12"}, "timezone"),
        ({"objects": []}, "objects"),
        ({"objects": [{"label": "keys", "confidence": 0.7, "bbox": [0.9, 0.5, 0.2, 0.1], "event": "placed"}]}, "inside the frame"),
        ({"objects": [{"label": "keys", "confidence": 1.4, "bbox": [0.1, 0.1, 0.1, 0.1], "event": "placed"}]}, "confidence"),
        ({"objects": [{"label": "keys", "confidence": 0.7, "bbox": [0.1, 0.1, 0.1, 0.1], "event": "lost"}]}, "event"),
        ({"observation_id": "not-a-uuid"}, "observation_id"),
    ],
)
def test_bad_observations_are_rejected_with_the_field_named(over, problem):
    with pytest.raises(ValidationError) as exc:
        Observation.model_validate(payload(**over))
    assert problem in str(exc.value)


def test_text_chunks_from_structured_labels():
    obs = Observation.model_validate(payload(objects=[
        {"label": "keys", "confidence": 0.71, "bbox": [0.4, 0.5, 0.1, 0.1], "event": "placed"},
        {"label": "wallet", "confidence": 0.41, "bbox": [0.1, 0.5, 0.1, 0.1], "event": "seen"},
        {"label": "phone", "confidence": 0.9, "bbox": [0.6, 0.5, 0.1, 0.1], "event": "picked_up"},
    ]))
    chunks = text_chunks(obs, obs.objects, "", SETTINGS)
    assert ("scene", "Camera saw the kitchen counter") in chunks
    assert ("item", "keys: kitchen counter") in chunks
    assert ("item", "wallet: kitchen counter (maybe)") in chunks  # below maybe_confidence
    assert ("item", "phone: picked up from the kitchen counter") in chunks
    assert summary(obs, obs.objects) == "keys, phone, wallet on the kitchen counter"


def test_future_timestamps_are_caught():
    assert too_far_ahead(datetime.now(timezone.utc) + timedelta(minutes=6))
    assert not too_far_ahead(datetime.now(timezone.utc) + timedelta(minutes=1))


# ---------------------------------------------------------------- where a passive sighting was seen

class FakeGeo:
    """Resolves a Wi-Fi scan to the Reitz Union (what Geolocator.locate returns)."""
    def __init__(self):
        self.scans = []

    async def locate(self, aps):
        self.scans.append(aps)
        from app.geo import Location
        return Location(29.6465, -82.3475, 25.0, "wifi") if aps else None


class PositionsDB:
    def __init__(self, recent=None):
        self.recent, self.asked = recent, []

    async def household_of_device(self, device_id):
        return None

    async def recent_position(self, device_ids, minutes):
        self.asked.append((device_ids, minutes))
        return self.recent


def locate(observation, db, geo=None, settings=SETTINGS):
    import asyncio
    from app.observations import observation_location
    return asyncio.run(observation_location(geo or FakeGeo(), db, "cam@clients", observation, settings))


def test_a_sightings_own_wifi_scan_places_it():
    geo = FakeGeo()
    obs = Observation.model_validate(payload(location={"wifi": [{"bssid": "aa:bb:cc:dd:ee:ff", "signal": -60},
                                                                {"bssid": "11:22:33:44:55:66", "signal": -70}]}))
    loc = locate(obs, PositionsDB(recent=(1.0, 2.0, 5.0)), geo)
    assert (loc.lat, loc.lng, loc.source) == (29.6465, -82.3475, "wifi") and len(geo.scans[0]) == 2


def test_without_its_own_location_a_sighting_takes_the_wearers_last_position():
    db = PositionsDB(recent=(29.65, -82.34, 40.0))
    loc = locate(Observation.model_validate(payload()), db)
    assert (loc.lat, loc.lng, loc.accuracy_m, loc.source) == (29.65, -82.34, 40.0, "recent")
    assert db.asked == [(["cam@clients"], 10.0)]
    assert locate(Observation.model_validate(payload()), PositionsDB(recent=None)) is None
    off = Settings(_env_file=None, location_reuse_minutes=0)
    assert locate(Observation.model_validate(payload()), PositionsDB(recent=(1.0, 2.0, 3.0)), settings=off) is None


def test_a_device_position_works_without_wifi_and_bad_locations_are_refused():
    loc = locate(Observation.model_validate(payload(location={"lat": 29.6, "lng": -82.3, "accuracy_m": 15})),
                 PositionsDB())
    assert (loc.lat, loc.lng) == (29.6, -82.3)
    for bad in ({"lat": 120, "lng": 0}, {"wifi": [{"bssid": "x", "signal": 10}]}, {"wifi": [{"signal": -50}]}):
        with pytest.raises(ValidationError):
            Observation.model_validate(payload(location=bad))


# ---------------------------------------------------------------- the public map

class PublicDB(PositionsDB):
    """A household that shares, whose every position falls in one public place."""
    def __init__(self, share=True, kind="public"):
        super().__init__()
        self.share, self.kind, self.stats = share, kind, []

    async def household_of_device(self, device_id):
        return ("hh-1", self.share)

    async def places_for(self, household_id):
        from app.db_places import Place
        return [Place("p-1", household_id, "Reitz Union", self.kind, 29.6465, -82.3475, 100.0)]

    async def set_location(self, table, row_id, lat, lng, accuracy_m, place_id):
        pass

    async def add_public_stats(self, household_id, lat, lng, kind, items, day):
        self.stats.append((household_id, round(lat, 4), round(lng, 4), kind, items))


def after(db, new_items):
    import asyncio
    from app.routers.observations import _after_observation
    obs = Observation.model_validate(payload(location={"lat": 29.6465, "lng": -82.3475, "accuracy_m": 20}))
    asyncio.run(_after_observation(db, None, FakeGeo(), None, SETTINGS, "cam@clients", "m-1", obs, b"", "image/jpeg",
                                   new_items))


def test_a_new_sighting_at_a_public_place_counts_on_the_public_map():
    db = PublicDB()
    after(db, ["keys"])
    assert db.stats == [("hh-1", 29.6465, -82.3475, "left", ["keys"])]


@pytest.mark.parametrize("db, new_items", [
    (PublicDB(), []),  # a merged refresh of the same memory
    (PublicDB(kind="private"), ["keys"]),
    (PublicDB(share=False), ["keys"]),
])
def test_refreshes_private_places_and_non_sharing_households_stay_off_the_public_map(db, new_items):
    after(db, new_items)
    assert db.stats == []
