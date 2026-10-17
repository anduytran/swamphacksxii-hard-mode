"""Eval pipeline plumbing: manifest loading, label semantics, stream recorder, contact sheet."""

import json

import cv2
import numpy as np
import pytest

from memory_engine.presets import PRESETS
from scripts.contact_sheet import contact_sheet
from scripts.eval_run import category, load_manifest, scored
from scripts.record_stream import record


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clips" / "a.mp4"
    path.parent.mkdir()
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    return path


def test_manifest_drops_placeholders_skips_missing_and_accepts_bom(tmp_path, clip):
    manifest = {"name": "t", "clips_dir": str(clip.parent), "clips": [
        {"id": "a", "file": "a.mp4", "windows": [[0, 1, "placed"]], "walk_away": None, "placed_at": 0.5},
        {"id": "b", "file": "missing.mp4"}]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest), encoding="utf-8-sig")  # with BOM, like PowerShell writes it
    m = load_manifest(path)
    assert [c["id"] for c in m["clips"]] == ["a"]
    assert "walk_away" not in m["clips"][0] and m["clips"][0]["placed_at"] == 0.5
    assert m["clips"][0]["useful"] == ["placed", "clutter"] and m["prompt"] == "bunch of keys"


def test_label_semantics():
    c = {"windows": [[0, 1, "hand"], [1.2, 2, "absent"], [2, 3, "unscored_far"], [3, 4, "pickup"]]}
    assert category(c, 500) == "hand" and category(c, 1100) is None
    assert scored("hand") and not scored("absent") and not scored("unscored_far") and not scored("pickup")
    assert not scored(None)


def test_record_from_file_paces_and_writes_report(tmp_path, clip):
    info = record(str(clip), str(tmp_path / "out"), "t", seconds=5)
    assert info["frames"] == 20 and info["resolution"] == [64, 48]
    assert 15 <= info["measured_fps"] <= 25            # replayed at the file's own ~20 fps, not as fast as possible
    assert (tmp_path / "out" / "t.mp4").exists() and (tmp_path / "out" / "t.json").exists()
    cap = cv2.VideoCapture(str(tmp_path / "out" / "t.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20


def test_contact_sheet(tmp_path, clip):
    out = contact_sheet(clip, every=0.25, out=tmp_path / "s.jpg")
    assert cv2.imread(str(out)) is not None


def test_presets_keep_efficient_as_default_and_candidates_have_distractors():
    from memory_engine.presets import DEFAULT_PRESET

    assert DEFAULT_PRESET == "efficient" and PRESETS["efficient"]["distractors"] == ()
    assert PRESETS["efficient"]["imgsz"] == 480 and PRESETS["efficient"]["model"] == "yolov8m-worldv2.pt"
    for name in ("m480d", "m640d", "m960d", "x960d"):
        assert "laptop keyboard" in PRESETS[name]["distractors"]
