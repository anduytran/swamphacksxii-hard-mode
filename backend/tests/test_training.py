"""Teaching the pipeline a household's items: the dashboard side and what the pipeline reads."""

import json
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import asyncpg
import pytest

from app.db_training import MAX_SAMPLES_PER_ITEM, MODEL_PART_BYTES, TrainedModel, TrainingItem, TrainingSample
from tests.test_api import CAREGIVER, DEVICE, FakeDB, client  # noqa: F401  (client is a fixture)

HOUSEHOLD = str(uuid.uuid4())
OTHER = str(uuid.uuid4())


class FakeTrainingDB(FakeDB):
    """FakeDB plus households, membership and the training tables, in memory."""

    def __init__(self):
        super().__init__()
        self.training: dict[str, tuple[str, TrainingItem]] = {}  # item id -> (household, item)
        self.sample_images: dict[str, tuple[bytes, str]] = {}

    async def member_role(self, household_id, user_sub):
        return "owner" if household_id == HOUSEHOLD and user_sub == "auth0|carer" else None

    async def household_of_device(self, device_id):
        return (HOUSEHOLD, False) if device_id == "dev1@clients" else None

    paired = True  # whether HOUSEHOLD has a wearable

    async def household_device_ids(self, household_id):
        return ["dev1@clients"] if household_id == HOUSEHOLD and self.paired else []

    async def training_bytes(self, household_id):
        return sum(len(img) for img, _ in self.sample_images.values())

    async def training_items(self, household_id):
        return sorted((i for h, i in self.training.values() if h == household_id), key=lambda i: i.name)

    async def count_training_items(self, household_id):
        return len(await self.training_items(household_id))

    async def create_training_item(self, household_id, name, aliases):
        if any(h == household_id and i.name == name for h, i in self.training.values()):
            raise asyncpg.UniqueViolationError("duplicate")
        item = TrainingItem(str(uuid.uuid4()), name, aliases, datetime.now(timezone.utc))
        self.training[item.id] = (household_id, item)
        return item

    async def update_training_item(self, household_id, item_id, name, aliases):
        found = self.training.get(item_id)
        if not found or found[0] != household_id:
            return False
        if name is not None:
            found[1].name = name
        if aliases is not None:
            found[1].aliases = aliases
        return True

    async def delete_training_item(self, household_id, item_id):
        found = self.training.get(item_id)
        if not found or found[0] != household_id:
            return False
        del self.training[item_id]
        return True

    async def training_item_sample_count(self, household_id, item_id):
        found = self.training.get(item_id)
        return len(found[1].samples) if found and found[0] == household_id else None

    async def add_training_sample(self, item_id, image, image_type, width, height, bbox, annotation="present"):
        sample = TrainingSample(str(uuid.uuid4()), item_id, bbox, width, height, datetime.now(timezone.utc), annotation)
        self.training[item_id][1].samples.append(sample)
        self.sample_images[sample.id] = (image, image_type)
        return sample

    async def delete_training_sample(self, household_id, sample_id):
        for h, item in self.training.values():
            for s in item.samples:
                if s.id == sample_id and h == household_id:
                    item.samples.remove(s)
                    return True
        return False

    async def restore_training_sample(self, household_id, sample_id, bbox, annotation):
        for h, item in self.training.values():
            for sample in item.samples:
                if h == household_id and sample.id == sample_id and sample.annotation in ("excluded", "unreviewed"):
                    sample.annotation = annotation
                    sample.bbox = bbox
                    return True
        return False

    async def training_sample_image(self, sample_id):
        return self.sample_images.get(sample_id)

    # trained models, in memory
    models: dict
    parts: dict

    def _models(self):
        if not hasattr(self, "models_store"):
            self.models_store, self.parts_store = {}, {}
        return self.models_store, self.parts_store

    async def create_model(self, household_id, version, files, metrics):
        models, _ = self._models()
        for mid in [m for m, (h, mod) in models.items() if h == household_id and not mod.complete]:
            del models[mid]
        model = TrainedModel(str(uuid.uuid4()), version, files, metrics, False, datetime.now(timezone.utc), None)
        models[model.id] = (household_id, model)
        return model.id

    async def get_model(self, household_id, model_id):
        found = self._models()[0].get(model_id)
        return found[1] if found and found[0] == household_id else None

    async def current_model(self, household_id):
        done = [m for h, m in self._models()[0].values() if h == household_id and m.complete]
        return max(done, key=lambda m: m.completed_at) if done else None

    async def put_model_part(self, model_id, name, part, data):
        self._models()[1][(model_id, name, part)] = data

    async def model_file(self, model_id, name):
        parts = self._models()[1]
        return b"".join(parts[k] for k in sorted(k for k in parts if k[0] == model_id and k[1] == name))

    async def model_part(self, model_id, name, part):
        return self._models()[1].get((model_id, name, part))

    async def complete_model(self, household_id, model_id):
        model = self._models()[0][model_id][1]
        model.complete, model.completed_at = True, datetime.now(timezone.utc)


@pytest.fixture
def api(client):  # noqa: F811
    client.app_state.db = FakeTrainingDB()
    return client


def items_url(household=HOUSEHOLD):
    return f"/api/v1/households/{household}/training/items"


def add_photo(api, item_id, bbox="[0.1, 0.2, 0.5, 0.4]", content_type="image/jpeg"):
    return api.post(
        f"{items_url()}/{item_id}/samples",
        headers=CAREGIVER,
        files={"image": ("k.jpg", b"\xff\xd8jpeg", content_type)},
        data={"bbox": bbox, "width": "640", "height": "480"},
    )


def test_members_add_items_and_photos(api):
    resp = api.post(items_url(), headers=CAREGIVER, json={"name": " Car  Keys ", "aliases": ["Keys", "fob", "car keys"]})
    assert resp.status_code == 201
    item = resp.json()
    assert item["name"] == "car keys" and item["aliases"] == ["keys", "fob"]  # its own name isn't an alias

    photo = add_photo(api, item["id"])
    assert photo.status_code == 201
    assert photo.json()["bbox"] == [0.1, 0.2, 0.5, 0.4]

    (listed,) = api.get(items_url(), headers=CAREGIVER).json()["items"]
    (sample,) = listed["samples"]
    image = api.get(sample["image_url"])  # a signed link: no token needed
    assert image.status_code == 200 and image.content == b"\xff\xd8jpeg"


def test_present_absent_and_unsubmitted_photos_stay_distinct(api):
    item = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    item_id = item["id"]
    absent = api.post(
        f"{items_url()}/{item_id}/samples", headers=CAREGIVER,
        files={"image": ("empty.jpg", b"\xff\xd8empty", "image/jpeg")},
        data={"annotation": "absent", "width": "640", "height": "480"},
    )
    assert absent.status_code == 201
    assert absent.json()["annotation"] == "absent" and absent.json()["bbox"] is None
    assert add_photo(api, item_id).status_code == 201  # legacy positive uploads still work
    samples = api.get(items_url(), headers=CAREGIVER).json()["items"][0]["samples"]
    assert [s["annotation"] for s in samples] == ["absent", "present"]
    assert samples[1]["bbox"] == [0.1, 0.2, 0.5, 0.4]
    # Skip makes no API request, so there is no third sample or inferred absence.
    assert len(samples) == 2
    exported = api.get("/api/v1/training/dataset", headers=DEVICE).json()["samples"]
    assert exported[0]["yolo"] == "" and exported[0]["bbox_xyxy_px"] is None
    assert exported[1]["yolo"].startswith("0 ")
    assert api.get("/api/v1/training/vocabulary", headers=DEVICE).json()["items"][0]["samples"] == 1


def test_absent_and_present_upload_validation(api):
    item_id = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()["id"]
    def upload(data):
        return api.post(f"{items_url()}/{item_id}/samples", headers=CAREGIVER,
                        files={"image": ("photo.jpg", b"\xff\xd8jpeg", "image/jpeg")}, data=data)
    assert upload({"annotation": "absent", "bbox": "[0,0,1,1]", "width": "640", "height": "480"}).status_code == 422
    assert upload({"annotation": "present", "width": "640", "height": "480"}).status_code == 422
    assert upload({"annotation": "skipped", "width": "640", "height": "480"}).status_code == 422
    assert api.get(items_url(), headers=CAREGIVER).json()["items"][0]["samples"] == []


def test_excluded_photo_is_retained_but_not_exported_until_boxed(api):
    item_id = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()["id"]
    response = api.post(f"{items_url()}/{item_id}/samples", headers=CAREGIVER,
                        files={"image": ("same.jpg", b"\xff\xd8same", "image/jpeg")},
                        data={"annotation": "excluded", "width": "640", "height": "480"})
    assert response.status_code == 201
    sample = response.json()
    assert sample["bbox"] is None
    assert api.get("/api/v1/training/dataset", headers=DEVICE).json()["samples"] == []
    version_before = api.get("/api/v1/training/vocabulary", headers=DEVICE).json()["version"]
    assert api.get(items_url(), headers=CAREGIVER).json()["items"][0]["samples"][0]["annotation"] == "excluded"
    assert api.patch(f"/api/v1/households/{OTHER}/training/samples/{sample['id']}",
                     headers=CAREGIVER, json={"bbox": [0.1, 0.2, 0.5, 0.4]}).status_code == 404
    assert api.patch(f"/api/v1/households/{HOUSEHOLD}/training/samples/{sample['id']}",
                     headers=CAREGIVER, json={"bbox": [0.1, 0.2, 0.5, 0.4]}).status_code == 200
    exported = api.get("/api/v1/training/dataset", headers=DEVICE).json()["samples"]
    assert len(exported) == 1 and exported[0]["annotation"] == "present"
    assert api.get("/api/v1/training/vocabulary", headers=DEVICE).json()["version"] != version_before


def test_closed_unlabeled_photo_survives_and_is_not_training_data(api):
    item_id = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()["id"]
    response = api.post(f"{items_url()}/{item_id}/samples", headers=CAREGIVER,
                        files={"image": ("later.jpg", b"\xff\xd8later", "image/jpeg")},
                        data={"annotation": "unreviewed", "width": "640", "height": "480"})
    assert response.status_code == 201
    sample_id = response.json()["id"]
    assert api.get(items_url(), headers=CAREGIVER).json()["items"][0]["samples"][0]["annotation"] == "unreviewed"
    assert api.get("/api/v1/training/dataset", headers=DEVICE).json()["samples"] == []
    restored = api.patch(f"/api/v1/households/{HOUSEHOLD}/training/samples/{sample_id}",
                         headers=CAREGIVER, json={"annotation": "absent"})
    assert restored.status_code == 200
    assert api.get("/api/v1/training/dataset", headers=DEVICE).json()["samples"][0]["annotation"] == "absent"


def test_duplicate_names_and_strangers_are_refused(api):
    assert api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).status_code == 201
    assert api.post(items_url(), headers=CAREGIVER, json={"name": "Keys"}).status_code == 409
    assert api.get(items_url(OTHER), headers=CAREGIVER).status_code == 404  # not their household
    assert api.get(items_url(), headers=DEVICE).status_code == 403  # devices use the pipeline endpoints
    assert api.post(items_url(), headers=CAREGIVER, json={"name": "   "}).status_code == 422


@pytest.mark.parametrize(
    "bbox", ["[0.1, 0.2, 0.5]", "[0.8, 0.2, 0.5, 0.4]", "[0.1, 0.2, 0.001, 0.4]", "not json", "[-0.1, 0, 0.5, 0.5]"]
)
def test_boxes_must_sit_inside_the_photo(api, bbox):
    item = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    assert add_photo(api, item["id"], bbox).status_code == 422


def test_photo_rules(api):
    item = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    assert add_photo(api, item["id"], content_type="image/gif").status_code == 415
    assert add_photo(api, str(uuid.uuid4())).status_code == 404
    for _ in range(MAX_SAMPLES_PER_ITEM):
        assert add_photo(api, item["id"]).status_code == 201
    assert add_photo(api, item["id"]).status_code == 409


def test_rename_delete_photo_and_item(api):
    item = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    sample = add_photo(api, item["id"]).json()
    assert api.patch(f"{items_url()}/{item['id']}", headers=CAREGIVER, json={"name": "House Keys"}).status_code == 200
    assert api.delete(f"/api/v1/households/{HOUSEHOLD}/training/samples/{sample['id']}", headers=CAREGIVER).json() == {"deleted": True}
    assert api.delete(f"{items_url()}/{item['id']}", headers=CAREGIVER).status_code == 200
    assert api.get(items_url(), headers=CAREGIVER).json()["items"] == []


def test_pipeline_reads_vocabulary_and_dataset(api):
    keys = api.post(items_url(), headers=CAREGIVER, json={"name": "keys", "aliases": ["key ring"]}).json()
    api.post(items_url(), headers=CAREGIVER, json={"name": "glasses"})
    add_photo(api, keys["id"], "[0.25, 0.5, 0.5, 0.25]")

    assert api.get("/api/v1/training/vocabulary", headers=CAREGIVER).status_code == 403
    vocab = api.get("/api/v1/training/vocabulary", headers=DEVICE).json()
    assert vocab["classes"] == ["glasses", "keys"]  # class id = position, the same in the dataset
    assert vocab["items"][1] == {"class_id": 1, "name": "keys", "aliases": ["key ring"], "samples": 1}

    data = api.get("/api/v1/training/dataset", headers=DEVICE).json()
    assert data["version"] == vocab["version"] and data["names"] == {"0": "glasses", "1": "keys"}
    (sample,) = data["samples"]
    assert sample["yolo"] == "1 0.500000 0.625000 0.500000 0.250000"  # class cx cy w h
    assert sample["bbox_xyxy_px"] == [160, 240, 480, 360]
    url = urlparse(sample["image_url"])
    assert url.scheme in ("http", "https") and url.netloc  # absolute: the pipeline host fetches it
    assert api.get(f"{url.path}?{url.query}").content == b"\xff\xd8jpeg"

    add_photo(api, keys["id"])
    assert api.get("/api/v1/training/vocabulary", headers=DEVICE).json()["version"] != vocab["version"]


def test_pipeline_needs_a_household(api):
    api.app_state.db.household_of_device = lambda device_id: _none()
    assert api.get("/api/v1/training/vocabulary", headers=DEVICE).status_code == 404


async def _none():
    return None


def test_dataset_json_is_plain(api):
    keys = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    add_photo(api, keys["id"])
    json.dumps(api.get("/api/v1/training/dataset", headers=DEVICE).json())  # serialisable as-is


def test_fetch_script_writes_a_yolo_folder(tmp_path, monkeypatch):
    import httpx

    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "scripts"))
    from fetch_training_set import write_dataset

    data = {
        "version": "abc",
        "names": {"0": "glasses", "1": "keys"},
        "samples": [{"id": "s1", "class_id": 1, "name": "keys", "image_url": "https://x/s1",
                     "yolo": "1 0.5 0.6 0.5 0.25", "bbox_xyxy_px": [160, 240, 480, 360]},
                    {"id": "s2", "class_id": 1, "name": "keys", "image_url": "https://x/s2",
                     "annotation": "absent", "yolo": "", "bbox_xyxy_px": None}],
    }
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "gone.jpg").write_bytes(b"old")  # deleted on the dashboard since
    http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"jpeg", headers={"content-type": "image/jpeg"})))
    assert write_dataset(data, tmp_path, http) == 2
    assert (tmp_path / "images" / "train" / "s1.jpg").read_bytes() == b"jpeg"
    assert (tmp_path / "labels" / "train" / "s1.txt").read_text() == "1 0.5 0.6 0.5 0.25\n"
    assert not (tmp_path / "labels" / "train" / "s2.txt").exists()
    assert not (tmp_path / "images" / "train" / "s2.jpg").exists()
    assert (tmp_path / "images" / "negatives" / "s2.jpg").read_bytes() == b"jpeg"
    assert json.loads((tmp_path / "negatives.json").read_text()) == [{"image": "images/negatives/s2.jpg", "class_id": 1, "name": "keys"}]
    assert not (tmp_path / "images" / "train" / "gone.jpg").exists()
    yaml = (tmp_path / "data.yaml").read_text()
    assert 'names:\n  0: "glasses"\n  1: "keys"\n' in yaml and "train: images/train" in yaml
    assert json.loads((tmp_path / "prompts.json").read_text())[0]["bbox_xyxy"] == [160, 240, 480, 360]


def test_teaching_needs_a_wearable(api):
    item = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    api.app_state.db.paired = False
    assert api.post(items_url(), headers=CAREGIVER, json={"name": "glasses"}).status_code == 409
    assert add_photo(api, item["id"]).status_code == 409


def test_household_photo_storage_is_capped(api, monkeypatch):
    import app.routers.training as training

    monkeypatch.setattr(training, "MAX_TRAINING_BYTES", 10)
    item = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    assert add_photo(api, item["id"]).status_code == 201  # 6 bytes
    assert add_photo(api, item["id"]).status_code == 409  # 12 > 10



def publish(api, version, files):
    """The pipeline's upload: manifest, parts, complete. Returns the complete call's response."""
    import hashlib

    manifest = [{"name": n, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()]
    created = api.post("/api/v1/training/models", headers=DEVICE, json={"version": version, "files": manifest, "metrics": {"map50": 0.5}})
    assert created.status_code == 201, created.text
    mid = created.json()["id"]
    for name, data in files.items():
        for i in range(0, max(1, -(-len(data) // MODEL_PART_BYTES))):
            chunk = data[i * MODEL_PART_BYTES : (i + 1) * MODEL_PART_BYTES]
            r = api.put(f"/api/v1/training/models/{mid}/files/{name}/parts/{i}", headers=DEVICE, content=chunk)
            assert r.status_code == 204, r.text
    return api.post(f"/api/v1/training/models/{mid}/complete", headers=DEVICE)


def test_models_upload_in_parts_and_download_byte_for_byte(api):
    big = bytes(range(256)) * (MODEL_PART_BYTES // 256 * 2 + 1000)  # a bit over 2 parts
    done = publish(api, "v1", {"model-int8.onnx": big, "classes.json": b'{"classes": ["keys"]}'})
    assert done.status_code == 200 and done.json()["current"]

    current = api.get("/api/v1/training/models/current", headers=DEVICE).json()
    assert current["version"] == "v1"
    onnx = next(f for f in current["files"] if f["name"] == "model-int8.onnx")
    assert onnx["parts"] == 3
    got = b"".join(
        api.get(f"/api/v1/training/models/{current['id']}/files/model-int8.onnx/parts/{i}", headers=DEVICE).content
        for i in range(onnx["parts"])
    )
    assert got == big


def test_model_upload_rules(api):
    import hashlib

    assert api.get("/api/v1/training/models/current", headers=DEVICE).status_code == 404
    bad_name = {"version": "v1", "files": [{"name": "../evil", "size": 1, "sha256": "0" * 64}]}
    assert api.post("/api/v1/training/models", headers=DEVICE, json=bad_name).status_code == 422
    no_model = {"version": "v1", "files": [{"name": "classes.json", "size": 1, "sha256": "0" * 64}]}
    assert api.post("/api/v1/training/models", headers=DEVICE, json=no_model).status_code == 422
    huge = {"version": "v1", "files": [{"name": "model-int8.onnx", "size": 41 * 1024 * 1024, "sha256": "0" * 64}]}
    assert api.post("/api/v1/training/models", headers=DEVICE, json=huge).status_code == 422
    assert api.post("/api/v1/training/models", headers=CAREGIVER, json=no_model).status_code == 403

    # a wrong checksum never becomes current
    data = b"onnx bytes"
    created = api.post("/api/v1/training/models", headers=DEVICE,
                       json={"version": "v2", "files": [{"name": "model-int8.onnx", "size": len(data), "sha256": hashlib.sha256(b"other").hexdigest()}]}).json()
    api.put(f"/api/v1/training/models/{created['id']}/files/model-int8.onnx/parts/0", headers=DEVICE, content=data)
    assert api.put(f"/api/v1/training/models/{created['id']}/files/model-int8.onnx/parts/5", headers=DEVICE, content=data).status_code == 422
    assert api.post(f"/api/v1/training/models/{created['id']}/complete", headers=DEVICE).status_code == 422
    assert api.get("/api/v1/training/models/current", headers=DEVICE).status_code == 404
    # nor can its parts be downloaded while it's uploading
    assert api.get(f"/api/v1/training/models/{created['id']}/files/model-int8.onnx/parts/0", headers=DEVICE).status_code == 409


def test_teach_page_model_status(api):
    keys = api.post(items_url(), headers=CAREGIVER, json={"name": "keys"}).json()
    add_photo(api, keys["id"])
    status = api.get(f"/api/v1/households/{HOUSEHOLD}/training/model", headers=CAREGIVER).json()
    assert status["model"] is None and status["photos"] == 1 and not status["up_to_date"]

    version = api.get("/api/v1/training/vocabulary", headers=DEVICE).json()["version"]
    publish(api, version, {"model-int8.onnx": b"m"})
    status = api.get(f"/api/v1/households/{HOUSEHOLD}/training/model", headers=CAREGIVER).json()
    assert status["up_to_date"] and status["model"]["map50"] == 0.5

    add_photo(api, keys["id"])  # new photos: needs retraining
    assert not api.get(f"/api/v1/households/{HOUSEHOLD}/training/model", headers=CAREGIVER).json()["up_to_date"]
