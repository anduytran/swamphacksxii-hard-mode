"""End-to-end on the real phone clips with the real detector (GPU, ~20 s).

Skipped unless LASTSEEN_E2E=1, so the normal test run stays fast and model-free:
    $env:LASTSEEN_E2E=1; pytest -q tests/test_e2e_real.py

Scenario (a wearable session built from real frames at 2 fps):
keys on the chair -> look away -> keys on the table. Then: ask where the keys are,
look at the table again for guidance, say "found it".
"""

import os
from pathlib import Path

import cv2
import pytest

from memory_engine.api import LastSeen, resolve_label
from memory_engine.memory import iou

CLIPS = Path(__file__).resolve().parents[1] / "data" / "clips"
pytestmark = pytest.mark.skipif(
    os.environ.get("LASTSEEN_E2E") != "1" or not (CLIPS / "desk_b.mov").exists(),
    reason="set LASTSEEN_E2E=1 (needs the phone clips and the YOLO-World model)")

TABLE_KEYS = (125, 280, 255, 430)  # keys on the table in desk_b, read off a frame by eye


def frames(clip, start, end):
    cap = cv2.VideoCapture(str(CLIPS / clip))
    t, out = start, []
    while t <= end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            break
        out.append(img)
        t += 0.5
    cap.release()
    return out


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    ls = LastSeen(out_dir=str(out))
    chair, away, table = frames("desk_a.mov", 0, 1.7), frames("desk_a.mov", 1.9, 3.5), frames("desk_b.mov", 0, 6.7)
    t = 0
    for img in chair + away + table:
        ls.process_frame(img, t, wall_ms=t)
        t += 500
    ls.finish()
    yield ls, out, len(chair) + len(away), table
    ls.close()


def test_memories_were_created_automatically(session):
    ls, *_ = session
    assert len(ls.memories(resolve_label("keys"))) == 2   # chair, then table


def test_where_are_my_keys_returns_the_table(session):
    ls, out, table_start, _ = session
    mem = ls.latest_memory(resolve_label("my keys"))
    assert mem["video_ms"] >= table_start * 500             # evidence frame is from the table part
    assert iou(mem["bbox"], TABLE_KEYS) >= 0.3              # and the box is on the keys
    assert mem["place_status"] in ("likely_new_place", "unverified")
    assert Path(mem["image_file"]).exists()


def test_guidance_at_the_table(session):
    ls, _, _, table = session
    mem = ls.latest_memory(resolve_label("keys"))
    g = ls.guide(table[2], mem["memory_id"])
    assert g["scene_verdict"] == "same_scene"
    assert g["status"] in ("object_at_remembered_spot", "remembered_spot_in_view")
    assert g["direction"]["in_frame"]


def test_found_it_clears_the_memory(session):
    ls, out, *_ = session
    mem = ls.latest_memory(resolve_label("keys"))
    ls.found(mem["memory_id"])
    assert not Path(mem["image_file"]).exists()
    remaining = ls.memories(resolve_label("keys"))
    assert mem["memory_id"] not in [m["memory_id"] for m in remaining]


# --- documented failure mode: cluttered table ------------------------------

def _first_memory_ms(clip):
    ls = LastSeen(out_dir=str(Path(os.environ.get("TEMP", ".")) / "lastseen_clutter"))
    try:
        for i, img in enumerate(frames(clip, 0, 6.7)):
            if any(e["event"] == "created" for e in ls.process_frame(img, i * 500, wall_ms=i * 500)):
                return i * 500
        return None
    finally:
        ls.found(label=resolve_label("keys"))
        ls.close()


def test_cluttered_table_eventually_creates_a_memory():
    assert _first_memory_ms("desk_b.mov") is not None


@pytest.mark.xfail(reason="Known failure mode: keys tangled with a lanyard/cable score 0.02-0.06, right at "
                          "min_conf=0.05, so a memory needs ~5 s of the keys sitting in view. A quick put-down "
                          "and walk-away on clutter is missed. See README 'Cluttered table'.", strict=False)
def test_cluttered_table_memory_within_3_seconds():
    first = _first_memory_ms("desk_b.mov")
    assert first is not None and first <= 3000
