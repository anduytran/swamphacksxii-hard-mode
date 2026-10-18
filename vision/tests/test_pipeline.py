"""lastscene_pipeline (sync / train / detect from Dylan's script spec): pure logic, no GPU, no network."""

import json
import os
import uuid

import cv2
import numpy as np
import pytest

from lastscene_bridge.uploader import CONFLICT_NAMESPACE
from lastscene_pipeline import EXIT_CONFIG, EXIT_NETWORK, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline.common import Api, PipelineError, _explain, resolve_model_dir
from lastscene_pipeline.detect import ReloadingDetector, folder_frames, load_classes
from lastscene_pipeline.runtime import decode, letterbox
from lastscene_pipeline.sync import run_sync
from lastscene_pipeline.train import add_backgrounds, clean_backgrounds, photos_per_class, run_train, set_current, train_yaml
from tests.test_bridge import _queue_one, bus, make_uploader, server  # noqa: F401

VOCAB = {"household_id": "h1", "version": "v1", "classes": ["keys", "wallet"],
         "items": [{"class_id": 0, "name": "keys", "aliases": ["car keys"], "samples": 2},
                   {"class_id": 1, "name": "wallet", "aliases": [], "samples": 0}]}


def dataset(version="v1", ids=("a", "b")):
    return {"household_id": "h1", "version": version, "names": {"0": "keys", "1": "wallet"},
            "samples": [{"id": i, "class_id": 0, "name": "keys", "image_url": f"https://signed/{i}", "width": 64,
                         "height": 48, "bbox": [0.1, 0.1, 0.5, 0.5], "bbox_xyxy_px": [6, 5, 38, 29],
                         "yolo": "0 0.350000 0.350000 0.500000 0.500000"} for i in ids]}


class FakeApi:
    def __init__(self, vocab, data):
        self.vocab, self.data, self.calls = vocab, data, []

    def get_json(self, path):
        self.calls.append(path)
        return self.vocab if path.endswith("vocabulary") else self.data


class FakeHttp:
    def __init__(self, status=200):
        self.status, self.urls = status, []

    def get(self, url, timeout=None):
        self.urls.append(url)
        r = type("R", (), {})()
        r.status_code, r.headers = self.status, {"content-type": "image/jpeg"}
        r.content = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()
        return r


@pytest.fixture
def photos(monkeypatch):
    http = FakeHttp()
    monkeypatch.setattr("lastscene_pipeline.sync.requests.Session", lambda: http)
    return http


# --- sync ---------------------------------------------------------------------------------------
def test_sync_writes_the_spec_layout_then_skips_an_unchanged_version(tmp_path, photos):
    out = tmp_path / "dataset"
    assert run_sync(out, api=FakeApi(VOCAB, dataset()), log=lambda m: None) == EXIT_OK
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["a.jpg", "b.jpg"]
    assert (out / "labels" / "train" / "a.txt").read_text() == "0 0.350000 0.350000 0.500000 0.500000\n"
    yaml = (out / "data.yaml").read_text()
    assert "train: images/train\nval: images/train\nnames:\n  0: \"keys\"\n  1: \"wallet\"" in yaml  # photo-less items keep an id
    assert json.loads((out / "classes.json").read_text()) == VOCAB
    assert json.loads((out / "prompts.json").read_text())[0] == {"image": "images/train/a.jpg", "class_id": 0,
                                                                   "name": "keys", "bbox_xyxy": [6, 5, 38, 29]}
    assert (out / "version").read_text() == "v1"
    api = FakeApi(VOCAB, dataset())
    assert run_sync(out, api=api, log=lambda m: None) == EXIT_NOTHING and api.calls == ["/api/v1/training/vocabulary"]
    assert run_sync(out, force=True, api=FakeApi(VOCAB, dataset()), log=lambda m: None) == EXIT_OK


def test_sync_keeps_item_specific_absence_out_of_yolo_training(tmp_path, photos):
    out = tmp_path / "dataset"
    data = dataset(ids=("positive",))
    data["samples"].append({"id": "absent", "class_id": 0, "name": "keys",
                            "image_url": "https://signed/absent", "width": 64, "height": 48,
                            "bbox": None, "bbox_xyxy_px": None, "annotation": "absent", "yolo": ""})
    assert run_sync(out, api=FakeApi(VOCAB, data), log=lambda m: None) == EXIT_OK
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["positive.jpg"]
    assert sorted(p.name for p in (out / "images" / "negatives").iterdir()) == ["absent.jpg"]
    assert json.loads((out / "negatives.json").read_text())[0]["class_id"] == 0
    assert sorted(p.name for p in (out / "labels" / "train").iterdir()) == ["positive.txt"]


def test_sync_removes_photos_deleted_on_the_dashboard(tmp_path, photos):
    out = tmp_path / "dataset"
    run_sync(out, api=FakeApi(VOCAB, dataset()), log=lambda m: None)
    v2 = dict(VOCAB, version="v2")
    run_sync(out, api=FakeApi(v2, dataset("v2", ids=("b",))), log=lambda m: None)
    assert [p.name for p in (out / "images" / "train").iterdir()] == ["b.jpg"]
    assert [p.name for p in (out / "labels" / "train").iterdir()] == ["b.txt"]


def test_sync_failure_never_leaves_a_version_behind(tmp_path, photos):
    out = tmp_path / "dataset"
    run_sync(out, api=FakeApi(VOCAB, dataset()), log=lambda m: None)
    photos.status = 403  # signed link expired mid-sync
    with pytest.raises(PipelineError) as e:
        run_sync(out, force=True, api=FakeApi(VOCAB, dataset()), log=lambda m: None)
    assert e.value.code == EXIT_NETWORK and not (out / "version").exists()  # next run re-downloads


def test_sync_skip_leaves_an_items_photos_out_but_keeps_its_class_id(tmp_path, photos):
    out = tmp_path / "dataset"
    data = dataset()
    data["samples"][1].update(class_id=1, name="Wallet", yolo="1 0.350000 0.350000 0.500000 0.500000")
    assert run_sync(out, api=FakeApi(VOCAB, data), log=lambda m: None, skip=["wallet"]) == EXIT_OK
    assert [p.name for p in (out / "images" / "train").iterdir()] == ["a.jpg"]
    assert [p["name"] for p in json.loads((out / "prompts.json").read_text())] == ["keys"]
    assert '1: "wallet"' in (out / "data.yaml").read_text() and (out / "version").read_text() == "v1"


def test_api_errors_map_to_spec_exit_codes():
    def r(code, location=None):
        return type("R", (), {"status_code": code, "text": "x", "headers": {"Location": location}})()
    assert _explain("/api/v1/training/vocabulary", r(404))[0] == EXIT_CONFIG  # not in a household yet
    assert "household" in _explain("/api/v1/training/vocabulary", r(404))[1]
    assert _explain("/x", r(403))[0] == EXIT_CONFIG and "scope" in _explain("/x", r(403))[1]
    assert _explain("/x", r(503))[0] == EXIT_NETWORK and _explain("/x", r(429))[0] == EXIT_NETWORK
    assert "www" in _explain("/x", r(308, "https://www.lastscene.photo/x"))[1]


def test_api_without_credentials_is_a_config_error(monkeypatch):
    with pytest.raises(PipelineError) as e:
        Api(env={})
    assert e.value.code == EXIT_CONFIG


# --- train ---------------------------------------------------------------------------------------
def test_train_refuses_without_sync_output_and_exits_3_without_photos(tmp_path, photos):
    with pytest.raises(PipelineError) as e:
        run_train(tmp_path / "missing" / "data.yaml", "yolo11s.pt", 1, 64, tmp_path / "models", log=lambda m: None)
    assert e.value.code == EXIT_CONFIG
    out = tmp_path / "dataset"
    run_sync(out, api=FakeApi(dict(VOCAB, version="empty"), dataset("empty", ids=())), log=lambda m: None)
    assert run_train(out / "data.yaml", "yolo11s.pt", 1, 64, tmp_path / "models", log=lambda m: None) == EXIT_NOTHING
    assert not (tmp_path / "models" / "current").exists()


def test_photo_counts_and_current_pointer(tmp_path, photos):
    out = tmp_path / "dataset"
    run_sync(out, api=FakeApi(VOCAB, dataset()), log=lambda m: None)
    assert photos_per_class(out, 2) == [2, 0]
    models = tmp_path / "models"
    (models / "v1").mkdir(parents=True)
    set_current(models, "v1")
    assert resolve_model_dir(models / "current") == models / "v1"
    (models / "v2").mkdir()
    set_current(models, "v2")
    assert resolve_model_dir(models / "current") == models / "v2"
    assert resolve_model_dir(models / "nope") is None


# --- runtime decode -------------------------------------------------------------------------------
def test_decode_maps_raw_yolo_output_back_to_frame_pixels():
    image = np.zeros((360, 640, 3), np.uint8)
    tensor, tf = letterbox(image, 320)  # gain 0.5, 80 px padding top and bottom
    assert tensor.shape == (1, 3, 320, 320) and tf == (0.5, 0, 70)
    out = np.zeros((1, 6, 3), np.float32)       # 2 classes: cx, cy, w, h, keys, wallet
    out[0, :, 0] = [100, 120, 40, 20, 0.9, 0.1]  # keys at (80..120, 110..130) letterboxed
    out[0, :, 1] = [101, 121, 40, 20, 0.6, 0.0]  # overlapping keys: suppressed by NMS
    out[0, :, 2] = [200, 160, 40, 40, 0.1, 0.2]  # wallet below the threshold
    (d,) = decode(out, tf, image.shape, ["keys", "wallet"], confidence=0.25)
    assert d.label == "keys" and d.conf == 0.9 and d.bbox == (160, 80, 240, 120)
    with pytest.raises(ValueError):
        decode(out, tf, image.shape, ["keys"])  # class map mismatch must not load silently


# --- detect ---------------------------------------------------------------------------------------
def test_folder_frames_use_file_times(tmp_path):
    for name, stamp in [("b.jpg", 1_800_000_010), ("a.jpg", 1_800_000_000)]:
        f = tmp_path / name
        cv2.imwrite(str(f), np.zeros((10, 10, 3), np.uint8))
        os.utime(f, (stamp, stamp))
    frames = list(folder_frames(tmp_path, 2))
    assert [f.observed_ms for f in frames] == [1_800_000_000_000, 1_800_000_010_000]  # a.jpg, then b.jpg
    assert [f.video_ms for f in frames] == [0, 10_000]


def test_classes_come_from_the_model_then_the_dataset(tmp_path):
    assert load_classes(None, tmp_path) == ["keys"]
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "classes.json").write_text(json.dumps(VOCAB))
    assert load_classes(None, tmp_path) == ["keys", "wallet"]
    model = tmp_path / "m"
    model.mkdir()
    (model / "classes.json").write_text(json.dumps({"classes": ["glasses"]}))
    assert load_classes(model, tmp_path) == ["glasses"]


def test_detector_reloads_when_current_moves_and_survives_a_bad_model(tmp_path):
    models = tmp_path / "models"
    for v in ("v1", "v2", "bad"):
        (models / v).mkdir(parents=True)
    set_current(models, "v1")

    class Det:
        def __init__(self, d):
            if d.name == "bad":
                raise ValueError("broken")
            self.name = d.name

        def detect(self, image):
            return [self.name]

    logs = []
    rd = ReloadingDetector(models / "current", None, Det, every=0, log=logs.append)
    assert rd.detect(None) == ["v1"]
    set_current(models, "v2")
    assert rd.detect(None) == ["v2"] and "reloaded" in logs[-1]
    set_current(models, "bad")
    assert rd.detect(None) == ["v2"] and "failed to load" in logs[-1]


# --- uploader: 409 and per-post log lines -----------------------------------------------------------
def test_409_resends_under_a_new_stable_id_and_reports_each_post(tmp_path, bus, server):
    state, url = server
    up = make_uploader(url, tmp_path / "q")
    lines = []
    up.on_result = lambda summary, oid, status, http: lines.append((summary, oid, status, http))
    oid, _ = _queue_one(up, bus)
    state.forced = [409]
    (result,) = up.flush()
    assert result == (oid, "sent")
    (summary, new_id, status, http) = lines[0]
    assert new_id != oid and new_id in state.observations and oid not in state.observations
    assert summary == "keys:seen" and status == "sent" and http == 201
    assert any("409" in s for _, _, s in up.log)
    # derived from (old id, pairing code): a restart mid-retry picks the same new id
    assert new_id == str(uuid.uuid5(CONFLICT_NAMESPACE, f"{oid}|{up.pairing_code}"))


def test_backgrounds_are_unlabeled_copies_and_join_training_only(tmp_path):
    camera, coco, dataset = tmp_path / "camera", tmp_path / "coco", tmp_path / "dataset"
    for folder, n in ((camera, 2), (coco, 5)):
        folder.mkdir()
        for i in range(n):
            cv2.imwrite(str(folder / f"{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
    (camera / "notes.txt").write_text("not an image")
    dataset.mkdir()
    coco_photos = sorted(coco.glob("*.jpg"))
    assert add_backgrounds(dataset, [camera], coco_photos, 3) == 5  # every camera frame + the first 3 COCO photos
    names = sorted(f.name for f in (dataset / "background").iterdir())
    assert names == [f"bg{i:04d}.jpg" for i in range(5)]  # no label files: Ultralytics reads them as "no item"
    assert add_backgrounds(dataset, [], coco_photos, 1) == 1  # a rerun replaces the folder
    assert add_backgrounds(dataset) == 0
    (dataset / "data.yaml").write_text('path: /x\ntrain: images/train\nval: images/train\nnames:\n  0: "pin"\n')
    text = train_yaml(dataset, dataset / "data.yaml").read_text()
    assert "train: [images/train, background]" in text and "val: images/train" in text and '0: "pin"' in text


def test_background_photos_never_contain_a_carryable_item(tmp_path):
    images, labels = tmp_path / "coco128/images/train2017", tmp_path / "coco128/labels/train2017"
    images.mkdir(parents=True), labels.mkdir(parents=True)
    rows = {"a": "0 .5 .5 .1 .1\n", "b": "0 .5 .5 .1 .1\n67 .2 .2 .1 .1\n", "c": None, "d": "41 .5 .5 .1 .1\n", "e": "\n"}
    for name, label in rows.items():  # 0 person, 67 cell phone, 41 cup
        cv2.imwrite(str(images / f"{name}.jpg"), np.zeros((8, 8, 3), np.uint8))
        if label is not None:
            (labels / f"{name}.txt").write_text(label)
    assert [f.stem for f in clean_backgrounds(images)] == ["a", "c", "e"]
