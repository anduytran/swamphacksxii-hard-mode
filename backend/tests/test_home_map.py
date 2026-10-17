from app.home_map import UNKNOWN, room_for


def test_locations_map_to_rooms():
    cases = {
        "kitchen counter": "kitchen",
        "hallway table": "hallway",
        "table by the front door": "hallway",
        "coat pocket": "hallway",
        "charger by the couch": "living",
        "armrest": "living",
        "nightstand": "bedroom",
        "case on the dresser": "bedroom",
        "bathroom shelf": "bathroom",
        "kitchen drawer": "kitchen",
        "dining table": "dining",
        "desk": "study",
    }
    for location, room in cases.items():
        assert room_for(location).id == room, location


def test_falls_back_to_scene_then_unknown():
    assert room_for("", "Keys on the wooden hallway table by the front door").id == "hallway"
    assert room_for("somewhere", None) is UNKNOWN
    assert room_for("indoors").id == "unknown"  # " door" must start a word


def test_item_spots_are_stable_and_inside_the_room():
    room = room_for("kitchen counter")
    a, b = room.spot("keys"), room.spot("keys")
    assert a == b
    assert room.x < a[0] < room.x + room.w and room.y < a[1] < room.y + room.h
