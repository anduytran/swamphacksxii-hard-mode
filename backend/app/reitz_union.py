"""The J. Wayne Reitz Union floor plan: the four levels SwampHacks uses, stacked.

Traced from the Reitz Union level maps (2025): each level's outline and its rooms, in the
maps' own coordinates (a 1024 x 1600 page per level), then scaled to floor-plan units. The
pages draw each level a little shifted, so every level is lined up on the orange elevator
(the same shaft on every floor); the dashboard stacks them, Ground at the bottom. Room
shapes are simplified to rectangles where the real ones are close to rectangular.

Meeting rooms are each their own room (G310–G330, 2101–2108, 2315–2365, 3305–3320), from the vector shapes and
room numbers in the 2025 level maps PDF: page points x 1024/792, y 1600/1224 (the 1024 x 1600 pages above), each
room's rectangle being the smallest green shape around its number (3305: the part of its block left of 3310).

Keywords: rooms are matched before corridors, and within rooms the earlier entry wins, so
specific places come first and the Grand Ballroom (where teams hack) takes the generic
"ballroom", "table" and "desk" last of all. Plain "restroom" or "hallway" means Level 2, next to the ballroom.
"""

from .rooms import Room

SCALE = 0.1  # map pixels -> floor-plan units
LEFT, TOP = 40, 200  # map pixels left of and above the drawings
GAP = 8  # floor-plan units between the building and "Elsewhere"
ELEVATOR = (726, 1111)  # the orange elevator on the Level 2 page; other levels line up to it

Px = tuple[float, float]


def _level(
    level: int,
    elevator: tuple[float, float],
    prefix: str,
    title: str,
    hall_words: tuple[str, ...],
    outline: list[Px],
    hall_area: tuple[float, float, float, float],
    rooms: list[tuple[str, str, tuple[float, float, float, float], tuple[str, ...]]],
    rooms_shaped: dict[str, list[Px]] | None = None,
) -> list[Room]:
    """One level: its corridors (the outline) and its rooms, moved so its orange elevator
    (at `elevator` on its page) sits where Level 2's does."""
    dx, dy = ELEVATOR[0] - elevator[0], ELEVATOR[1] - elevator[1]

    def pt(x: float, y: float) -> tuple[float, float]:
        return (round((x + dx - LEFT) * SCALE, 2), round((y + dy - TOP) * SCALE, 2))

    def box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        px, py = pt(x, y)
        return px, py, round(w * SCALE, 2), round(h * SCALE, 2)

    out = [Room(f"{prefix}_halls", title, *box(*hall_area), hall_words, tuple(pt(x, y) for x, y in outline), "hall", level)]
    for room_id, name, area, words in rooms:
        shape = tuple(pt(x, y) for x, y in (rooms_shaped or {}).get(room_id, []))
        out.append(Room(room_id, name, *box(*area), words, shape, "room", level))
    return out


ground = _level(
    0,
    (693, 1073),
    "ground",
    "Ground level",
    ("ground floor", "ground level", "basement"),
    [(597, 215), (925, 215), (925, 660), (953, 660), (953, 718), (912, 718), (912, 1480), (805, 1480),
     (805, 1540), (585, 1540), (585, 1470), (550, 1465), (595, 1250), (400, 1065), (400, 1040), (165, 1040),
     (165, 715), (372, 715), (372, 660), (597, 660)],
    (640, 1050, 115, 380),
    [
        ("game_room", "Game Room", (165, 715, 435, 325), ("game room", "bowling", "arcade", "pool table", "billiards", "ping pong", "games")),
        ("esports", "Gator Esports Center", (372, 660, 265, 55), ("esports", "gaming")),
        ("orange_brew", "Orange & Brew", (480, 1060, 150, 120), ("orange & brew", "orange and brew", "orange brew", "coffee", "cafe")),
        ("breezeway", "Breezeway", (597, 215, 323, 308), ("breezeway",)),
        ("career_ground", "Career Connections Center (ground)", (720, 523, 200, 137), ()),
        ("hotel", "Hotel Reception", (640, 738, 85, 75), ("hotel", "reception")),
        ("dining_ground", "Dining", (760, 718, 150, 95), ("dining",)),
        ("bike_repair", "SG Bike Repair", (848, 852, 62, 90), ("bike",)),
        ("senate", "Senate Chamber", (758, 1055, 154, 170), ("senate", "chamber")),
        ("room_g310", "Meeting Room G310", (776, 1226, 126, 84), ("g310",)),
        ("room_g315", "Meeting Room G315", (776, 1310, 126, 70), ("g315",)),
        ("room_g320", "Meeting Room G320", (776, 1381, 126, 102), ("g320",)),
        ("room_g325", "Meeting Room G325", (685, 1418, 78, 119), ("g325",)),
        ("room_g330", "Meeting Room G330", (580, 1418, 105, 119), ("g330",)),
        ("restrooms_ground", "Restrooms (ground)", (660, 912, 67, 90), ()),
    ],
    {"orange_brew": [(400, 1040), (635, 1040), (635, 1245), (590, 1215), (400, 1065)]},
)

level1 = _level(
    1,
    (756, 1088),
    "level1",
    "Level 1",
    ("first floor", "level 1", "level one"),
    [(662, 243), (973, 243), (973, 1235), (853, 1275), (865, 1545), (652, 1545), (660, 1290), (640, 1230),
     (520, 1135), (355, 1118), (355, 1050), (235, 1050), (235, 968), (205, 968), (205, 1088), (47, 1088),
     (47, 780), (235, 780), (235, 735), (662, 735)],
    (665, 1290, 185, 180),
    [
        ("info_desk", "Information Desk", (800, 1235, 55, 45), ("info desk", "information desk", "help desk", "front desk", "lost and found", "check-in", "check in", "registration")),
        ("constans", "Constans Theatre", (652, 1480, 213, 65), ("constans", "theatre", "theater")),
        ("bookstore", "UF Bookstore", (47, 800, 150, 270), ("bookstore", "book store")),
        ("career", "Career Connections Center", (662, 243, 311, 465), ("career",)),
        ("gatorwell", "GatorWell", (838, 737, 135, 195), ("gatorwell", "health")),
        ("admin", "Reitz Union Administration", (838, 935, 135, 165), ("administration", "admin office")),
        ("catering", "Catering", (838, 1100, 90, 85), ("catering",)),
        ("event_services", "Event Services", (845, 1190, 100, 45), ("event services",)),
        ("restrooms_1", "Restrooms (level 1)", (722, 900, 68, 130), ()),
        ("food_court", "Food Court", (240, 740, 390, 280), ("food court", "food", "starbucks", "lunch", "dinner", "breakfast", "snack", "chick-fil-a", "chick fil a", "panda", "subway", "kitchen", "counter")),
    ],
    {"bookstore": [(47, 780), (235, 780), (235, 968), (205, 968), (205, 1088), (47, 1088)]},
)

level2 = _level(
    2,
    ELEVATOR,
    "level2",
    "Level 2",
    ("second floor", "level 2", "level two", "hallway", "hall", "corridor", "elevator", "stairs", "bench", "lobby", "door", "coat"),
    [(633, 305), (945, 305), (940, 725), (933, 775), (933, 1080), (948, 1080), (948, 1500), (830, 1500),
     (830, 1553), (622, 1553), (622, 1505), (595, 1505), (595, 1300), (560, 1220), (475, 1145), (355, 1145),
     (355, 1080), (227, 1080), (227, 775), (633, 775)],
    (640, 725, 290, 50),
    [
        ("restrooms", "Restrooms", (768, 307, 125, 35), ("restroom", "bathroom", "sink", "toilet", "mirror")),
        ("auditorium", "Auditorium", (800, 778, 133, 210), ("auditorium",)),
        ("gallery", "The Gallery", (800, 990, 133, 90), ("gallery",)),
        ("student_engagement", "Student Engagement", (815, 1080, 130, 420), ("student engagement",)),
        ("leadership", "Brown Center for Leadership and Service", (480, 1085, 175, 150), ("leadership", "brown center")),
        ("copy_center", "SG Graphics and Copy Center", (688, 1445, 75, 45), ("copy center", "print", "printer", "graphics")),
        ("room_2101", "Meeting Room 2101", (590, 1309, 45, 47), ("2101",)),
        ("room_2102", "Meeting Room 2102", (590, 1356, 56, 65), ("2102",)),
        ("room_2103", "Meeting Room 2103", (590, 1421, 41, 87), ("2103",)),
        ("room_2104", "Meeting Room 2104", (631, 1442, 34, 35), ("2104",)),
        ("room_2106", "Meeting Room 2106", (643, 1492, 150, 60), ("2106",)),
        ("room_2108", "Meeting Room 2108", (771, 1441, 37, 36), ("2108",)),
        ("room_2345", "Meeting Room 2345", (436, 1081, 36, 67), ("2345",)),
        ("room_2350", "Meeting Room 2350", (398, 1081, 37, 62), ("2350",)),
        ("room_2315", "Meeting Room 2315", (482, 989, 70, 60), ("2315",)),
        ("room_2320", "Meeting Room 2320", (482, 929, 70, 60), ("2320",)),
        ("room_2325", "Meeting Room 2325", (482, 868, 70, 61), ("2325",)),
        ("room_2330", "Meeting Room 2330", (482, 806, 70, 61), ("2330",)),
        ("room_2335", "Meeting Room 2335", (398, 802, 62, 125), ("2335",)),
        ("room_2340", "Meeting Room 2340", (398, 927, 62, 121), ("2340",)),
        ("room_2355", "Meeting Room 2355", (283, 958, 115, 90), ("2355",)),
        ("room_2360", "Meeting Room 2360", (283, 896, 115, 62), ("2360",)),
        ("room_2365", "Meeting Room 2365", (283, 802, 115, 94), ("2365",)),
        ("restrooms_2", "Restrooms (level 2, centre)", (695, 960, 62, 72), ()),
        ("grand_ballroom", "Grand Ballroom", (673, 373, 229, 345), ("grand ballroom", "ballroom", "hacking", "hack", "team", "workstation", "laptop", "monitor", "charger", "outlet", "table", "desk", "chair")),
    ],
    {"leadership": [(475, 1080), (660, 1080), (660, 1300), (595, 1300), (560, 1220), (475, 1145)]},
)

level3 = _level(
    3,
    (736, 1105),
    "level3",
    "Level 3",
    ("third floor", "level 3", "level three"),
    [(228, 765), (692, 765), (692, 715), (948, 715), (948, 975), (963, 1070), (963, 1500), (843, 1500),
     (843, 1555), (630, 1555), (630, 1505), (603, 1505), (603, 1300), (560, 1210), (500, 1145), (430, 1112),
     (430, 1075), (228, 1075)],
    (230, 1052, 460, 22),
    [
        ("rion", "Rion Ballroom", (310, 787, 217, 265), ("rion", "stage", "judging", "judge", "demo", "presentation", "ceremony", "podium", "closing", "opening")),
        ("room_3305", "Meeting Room 3305", (816, 715, 59, 78), ("3305",)),
        ("room_3310", "Meeting Room 3310", (875, 715, 63, 78), ("3310",)),
        ("room_3315", "Meeting Room 3315", (815, 817, 123, 79), ("3315",)),
        ("room_3320", "Meeting Room 3320", (816, 896, 122, 79), ("3320",)),
        ("legal", "Student Legal Services", (692, 715, 124, 80), ("legal",)),  # to where the meeting rooms start
        ("reflection", "Reflection Room", (802, 975, 90, 65), ("reflection", "prayer", "quiet room")),
        ("greek_life", "Sorority and Fraternity Life", (810, 1040, 150, 300), ("sorority", "fraternity")),
        ("student_government", "Student Government", (603, 1300, 90, 200), ("student government",)),
        ("sg_finance", "Student Government Finance", (430, 1075, 180, 100), ("finance",)),
        ("restrooms_3", "Restrooms (level 3)", (705, 955, 65, 70), ()),
    ],
)

# Room numbers ("2360", "G315") are the most specific thing anyone says, so numbered rooms are matched first; the
# Grand Ballroom's generic words ("ballroom", "table", "desk") are matched last of all.
_rooms = (*ground, *level1, *level2, *level3)
REITZ_ROOMS: tuple[Room, ...] = (
    *(r for r in _rooms if r.id.startswith("room_")),
    *(r for r in _rooms if not r.id.startswith("room_") and r.id != "grand_ballroom"),
    *(r for r in _rooms if r.id == "grand_ballroom"),
)
# Levels 4 and Lower, anywhere outside, or no place said: beside the building, at ground level.
_right = max(x for r in _rooms for x, _ in r.polygon)
REITZ_UNKNOWN = Room("unknown", "Elsewhere", round(_right + GAP, 2), 10, 18, 120, ())
