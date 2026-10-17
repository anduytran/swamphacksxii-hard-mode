from app.home_map import UNKNOWN, floor_plan, room_for


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


def test_reitz_union_plan():
    plan = floor_plan("reitz_union")
    cases = {
        "hacking table in the grand ballroom": "grand_ballroom",
        "my team's table": "grand_ballroom",
        "the charger by my laptop": "grand_ballroom",
        "the stage in the Rion Ballroom": "rion",
        "food court table": "food_court",
        "Starbucks counter": "food_court",
        "pool table in the game room": "game_room",
        "registration desk": "info_desk",
        "meeting room 2340": "meeting_2315",
        "bench in the hallway": "level2_halls",
        "restroom sink": "restrooms",
        "somewhere on the third floor": "level3_halls",
    }
    for location, room in cases.items():
        assert room_for(location, plan=plan).id == room, location
    assert room_for("on the moon", plan=plan) is plan.unknown


def test_reitz_union_rooms_do_not_overlap_and_sit_on_their_level():
    plan = floor_plan("reitz_union")
    rooms = [r for r in plan.rooms if r.kind == "room"]
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            assert a.level != b.level or a.x + a.w <= b.x + 0.01 or b.x + b.w <= a.x + 0.01 or a.y + a.h <= b.y + 0.01 or b.y + b.h <= a.y + 0.01, (a.id, b.id)
    halls = [r for r in plan.rooms if r.kind == "hall"]
    assert [h.name for h in halls] == ["Ground level", "Level 1", "Level 2", "Level 3"]
    assert [h.level for h in halls] == [0, 1, 2, 3]
    for room in rooms:  # every room's area falls within its own level's outline box
        (hall,) = [h for h in halls if h.level == room.level]
        xs, ys = [x for x, _ in hall.polygon], [y for _, y in hall.polygon]
        assert min(xs) - 0.01 <= room.x and room.x + room.w <= max(xs) + 0.01, room.id
        assert min(ys) - 0.01 <= room.y and room.y + room.h <= max(ys) + 0.01, room.id
    assert plan.unknown.x > max(x for h in halls for x, _ in h.polygon)


def test_unknown_plan_falls_back_to_home():
    assert floor_plan("nope").id == "home"
