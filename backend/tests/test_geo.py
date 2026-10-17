import asyncio

from app.db_places import Place, cell_of
from app.geo import Location, distance_m, home_place_id, nearest_place, parse_wifi, place_label, resolve_location


def test_parse_wifi_keeps_valid_aps_strongest_first():
    raw = '[{"bssid": "AA:BB:CC:00:11:22", "signal": -70}, {"bssid": "bad"}, {"bssid": "aa:bb:cc:00:11:33", "signal": -40}]'
    aps = parse_wifi(raw)
    assert [a.bssid for a in aps] == ["aa:bb:cc:00:11:33", "aa:bb:cc:00:11:22"]
    assert parse_wifi("not json") == [] and parse_wifi(None) == []


def test_distance():
    assert abs(distance_m(29.6516, -82.3248, 29.6516, -82.3248)) < 1e-6
    assert 1500 < distance_m(29.6516, -82.3248, 29.6550, -82.3400) < 1600  # ~1.5 km across town


def place(pid, lat, lng, radius=75, name=None, visits=0):
    return Place(pid, "h", name, "private", lat, lng, radius, visits)


def test_nearest_place_respects_radius_and_accuracy():
    places = [place("a", 29.6516, -82.3248), place("b", 29.6550, -82.3400)]
    assert nearest_place(places, Location(29.6517, -82.3249, 10, "wifi")).id == "a"
    assert nearest_place(places, Location(29.6600, -82.3248, 10, "wifi")) is None  # ~900 m away
    assert nearest_place(places, Location(29.6600, -82.3248, 1200, "wifi")).id == "a"  # a vague fix can match


def test_home_is_the_most_visited_until_named():
    places = [place("a", 0, 0, visits=2), place("b", 1, 1, visits=9)]
    assert home_place_id(places) == "b"
    assert place_label(places[1], "b", 2) == "Home" and place_label(places[0], "b", 1) == "Place 1"
    places[0].name = "Home"
    assert home_place_id(places) == "a"


def test_public_cells_hide_exact_positions():
    lat, lng = cell_of(29.65163, -82.32481)
    assert (lat, lng) == cell_of(29.65199, -82.32401)  # same ~100 m cell
    assert abs(lat - 29.65163) < 0.001 and (lat, lng) != (29.65163, -82.32481)


def test_device_position_is_the_fallback():
    class NoGeo:
        async def locate(self, aps):
            return None

    loc = asyncio.run(resolve_location(NoGeo(), "[]", 29.65, -82.32, 100))
    assert (loc.lat, loc.lng, loc.source) == (29.65, -82.32, "device")
    assert asyncio.run(resolve_location(NoGeo(), None, None, None, None)) is None
    assert asyncio.run(resolve_location(NoGeo(), None, 123, 0, None)) is None  # invalid latitude
