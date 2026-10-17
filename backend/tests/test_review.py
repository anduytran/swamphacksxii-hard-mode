"""Teach's "Needs a look": suggested boxes from Gemini, the review queue and the caregiver's answers."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.box_suggest import BoxSuggestion, box_2d_to_xywh
from app.db_review import Review
from app.db_training import MAX_SAMPLES_PER_ITEM
from app.review import mentions
from tests.test_api import CAREGIVER, client  # noqa: F401  (client is a fixture)
from tests.test_training import HOUSEHOLD, OTHER, FakeTrainingDB, items_url

NOW = datetime.now(timezone.utc)


class FakeReviewDB(FakeTrainingDB):
    """FakeTrainingDB plus memory photos, detections and review answers, in memory."""

    def __init__(self):
        super().__init__()
        self.photos: dict[str, dict] = {}  # memory id -> {device, image, type, names, detections, at}
        self.answers: dict[tuple[str, str], Review] = {}
        self.sample_ids: dict[tuple[str, str], str] = {}
        self.suggest_calls: list[tuple[str, datetime]] = []

    def photo(self, names=(), detections=(), device="dev1@clients", image_type="image/jpeg", minutes_ago=5):
        mid = str(uuid.uuid4())
        self.photos[mid] = {"device": device, "image": b"\xff\xd8frame-" + mid.encode(), "type": image_type,
                            "names": list(names), "detections": list(detections), "at": NOW - timedelta(minutes=minutes_ago)}
        return mid

    async def review_detections(self, device_ids, low, high, hours):
        return [
            {"memory_id": mid, "created_at": p["at"], "label": label, "confidence": conf, "bbox": box, "width": 640, "height": 480}
            for mid, p in self.photos.items() if p["device"] in device_ids
            for label, conf, box in p["detections"] if low <= conf < high
        ]

    async def review_memory_photos(self, device_ids, hours):
        return [
            {"memory_id": mid, "created_at": p["at"], "names": p["names"], "boxed": [d[0] for d in p["detections"]]}
            for mid, p in self.photos.items() if p["device"] in device_ids and p["names"]
        ]

    async def reviews(self, household_id):
        return {k: r for k, r in self.answers.items() if self.training.get(k[0], (None,))[0] == household_id}

    async def decide_review(self, item_id, memory_id, decision, bbox, sample_id):
        old = self.answers.get((item_id, memory_id))
        self.answers[(item_id, memory_id)] = Review(item_id, memory_id, decision, bbox or (old.bbox if old else None),
                                                    old.note if old else "", bool(old and old.suggested))
        self.sample_ids[(item_id, memory_id)] = sample_id

    async def cache_suggestion(self, item_id, memory_id, bbox, note):
        old = self.answers.get((item_id, memory_id))
        if not (old and old.decision):
            self.answers[(item_id, memory_id)] = Review(item_id, memory_id, None, bbox, note, True)

    async def household_memory_image(self, memory_id, device_ids):
        p = self.photos.get(memory_id)
        return (p["image"], p["type"]) if p and p["device"] in device_ids else None

    async def suggest_calls_since(self, household_id, since):
        return sum(1 for h, at in self.suggest_calls if h == household_id and at > since)

    async def log_suggest_call(self, household_id):
        self.suggest_calls.append((household_id, datetime.now(timezone.utc)))


class FakeGemini:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    async def generate_json(self, system, contents, schema, *, fast=False):
        self.calls.append((system, contents, schema))
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture
def api(client):  # noqa: F811
    client.app_state.db = FakeReviewDB()
    client.app_state.ai.gemini = FakeGemini(BoxSuggestion(found=True, box_2d=[200, 100, 600, 500], quality=""))
    return client


@pytest.fixture
def keys(api):
    return api.post(items_url(), headers=CAREGIVER, json={"name": "keys", "aliases": ["key ring"]}).json()["id"]


def review_url(household=HOUSEHOLD):
    return f"/api/v1/households/{household}/training/review"


def suggest(api, item_id, household=HOUSEHOLD, **data):
    files = None if data else {"image": ("p.jpg", b"\xff\xd8jpeg", "image/jpeg")}
    return api.post(f"{items_url(household)}/{item_id}/suggest-box", headers=CAREGIVER, files=files, data=data or None)


# ---------------------------------------------------------------- box_2d


def test_box_2d_converts_to_normalised_xywh():
    assert box_2d_to_xywh([200, 100, 600, 500]) == [0.1, 0.2, 0.4, 0.4]  # [ymin, xmin, ymax, xmax] /1000
    assert box_2d_to_xywh([0, 0, 1000, 1000]) == [0, 0, 1, 1]
    assert box_2d_to_xywh([600, 500, 200, 100]) == [0.1, 0.2, 0.4, 0.4]  # corners reversed
    assert box_2d_to_xywh([-50, 900, 400, 1200]) == [0.9, 0, 0.1, 0.4]  # clamped to the photo
    assert box_2d_to_xywh([0, 0, 0, 0]) is None
    assert box_2d_to_xywh([100, 100, 105, 500]) is None  # a sliver
    assert box_2d_to_xywh([1, 2, 3]) is None


def test_item_names_match_whole_words_and_plurals():
    from app.db_training import TrainingItem

    keys = TrainingItem("1", "keys", ["key ring"], NOW)
    assert mentions(keys, "keys") and mentions(keys, "car keys") and mentions(keys, "key") and mentions(keys, "blue key ring")
    assert not mentions(keys, "keyboard") and not mentions(keys, "monkeys") and not mentions(keys, "ring")


# ---------------------------------------------------------------- suggest-box


def test_suggest_box_returns_gemini_box_as_xywh(api, keys):
    resp = suggest(api, keys)
    assert resp.status_code == 200
    assert resp.json() == {"bbox": [0.1, 0.2, 0.4, 0.4], "note": "", "status": "ok"}
    system, contents, _ = api.app_state.ai.gemini.calls[0]
    assert "keys" in system and "key ring" in system


def test_suggest_box_not_found_and_quality_note(api, keys):
    api.app_state.ai.gemini.answer = BoxSuggestion(found=False, box_2d=[0, 0, 0, 0], quality="blurry")
    assert suggest(api, keys).json() == {"bbox": None, "note": "blurry", "status": "none"}


def test_suggest_box_never_calls_gemini_when_vision_is_off(api, keys):
    api.settings.vision_provider = "none"
    assert suggest(api, keys).json()["status"] == "off"
    assert api.app_state.ai.gemini.calls == []


def test_suggest_box_fails_soft_on_quota(api, keys):
    api.app_state.ai.gemini.error = RuntimeError("429 RESOURCE_EXHAUSTED")
    resp = suggest(api, keys)
    assert resp.status_code == 200 and resp.json() == {"bbox": None, "note": "", "status": "failed"}


def test_suggest_box_is_rate_limited_per_household(api, keys):
    api.settings.suggest_box_per_hour = 2
    assert [suggest(api, keys).json()["status"] for _ in range(3)] == ["ok", "ok", "busy"]
    assert len(api.app_state.ai.gemini.calls) == 2


def test_suggest_box_is_for_members_only(api, keys):
    assert suggest(api, keys, household=OTHER).status_code == 404
    assert suggest(api, str(uuid.uuid4())).status_code == 404
    db = api.app_state.db
    stranger = db.photo(names=["keys"], device="someone-else@clients")
    assert suggest(api, keys, memory_id=stranger).status_code == 404  # another household's photo


# ---------------------------------------------------------------- queue


def test_queue_has_unsure_detections_and_unboxed_memory_photos(api, keys):
    db = api.app_state.db
    unsure = db.photo(detections=[("keys", 0.42, [0.1, 0.1, 0.2, 0.2])], minutes_ago=1)
    db.photo(detections=[("keys", 0.9, [0.1, 0.1, 0.2, 0.2])])  # sure: not asked
    db.photo(detections=[("wallet", 0.4, [0.1, 0.1, 0.2, 0.2])])  # not a taught item
    said = db.photo(names=["car keys"], minutes_ago=10)
    db.photo(names=["keys"], detections=[("keys", 0.95, [0.3, 0.3, 0.1, 0.1])])  # already boxed by a detector
    db.photo(names=["glasses"])
    db.photo(names=["keys"], device="someone-else@clients")  # not this household's wearable

    body = api.get(review_url(), headers=CAREGIVER).json()
    assert body["count"] == 2 and body["per_item"] == {keys: 2}
    first, second = body["candidates"]
    assert (first["memory_id"], first["kind"], first["bbox"], first["confidence"]) == (unsure, "detection", [0.1, 0.1, 0.2, 0.2], 0.42)
    assert first["image_url"].startswith(f"/api/v1/images/{unsure}?") and first["item_name"] == "keys"
    assert (second["memory_id"], second["kind"], second["bbox"], second["suggested"]) == (said, "memory", None, False)


def test_queue_band_comes_from_settings(api, keys):
    api.app_state.db.photo(detections=[("keys", 0.7, [0.1, 0.1, 0.2, 0.2])])
    assert api.get(review_url(), headers=CAREGIVER).json()["count"] == 0
    api.settings.review_confidence_max = 0.8
    assert api.get(review_url(), headers=CAREGIVER).json()["count"] == 1


def test_queue_is_for_members_only(api, keys):
    assert api.get(review_url(OTHER), headers=CAREGIVER).status_code == 404


def test_memory_card_suggestion_is_cached(api, keys):
    db = api.app_state.db
    said = db.photo(names=["keys"])
    assert suggest(api, keys, memory_id=said).json()["status"] == "ok"
    (card,) = api.get(review_url(), headers=CAREGIVER).json()["candidates"]
    assert card["bbox"] == [0.1, 0.2, 0.4, 0.4] and card["suggested"] is True


# ---------------------------------------------------------------- answers


def answer(api, keys, memory_id, decision, **extra):
    return api.post(review_url(), headers=CAREGIVER, json={"item_id": keys, "memory_id": memory_id, "decision": decision, **extra})


def test_yes_and_fix_become_training_samples(api, keys):
    db = api.app_state.db
    a = db.photo(detections=[("keys", 0.4, [0.1, 0.1, 0.2, 0.2])])
    b = db.photo(names=["keys"])

    yes = answer(api, keys, a, "yes", bbox=[0.1, 0.1, 0.2, 0.2], width=640, height=480)
    assert yes.status_code == 200 and yes.json()["decision"] == "yes"
    fix = answer(api, keys, b, "fix", bbox=[0.5, 0.5, 0.05, 0.05], width=1024, height=768)
    assert fix.json()["decision"] == "fixed"

    (item,) = api.get(items_url(), headers=CAREGIVER).json()["items"]
    assert [(s["bbox"], s["width"]) for s in item["samples"]] == [([0.1, 0.1, 0.2, 0.2], 640), ([0.5, 0.5, 0.05, 0.05], 1024)]
    assert db.sample_images[item["samples"][0]["id"]][0] == db.photos[a]["image"]  # the frame itself, copied
    assert db.answers[(keys, b)].decision == "fixed" and db.sample_ids[(keys, b)] == item["samples"][1]["id"]
    assert api.get(review_url(), headers=CAREGIVER).json()["count"] == 0
    assert answer(api, keys, a, "yes", bbox=[0.1, 0.1, 0.2, 0.2], width=640, height=480).status_code == 409  # once


def test_not_is_remembered_and_adds_no_sample(api, keys):
    db = api.app_state.db
    a = db.photo(detections=[("keys", 0.4, [0.1, 0.1, 0.2, 0.2])])
    resp = answer(api, keys, a, "not")
    assert resp.status_code == 200 and resp.json() == {"decision": "not", "sample": None}
    assert db.answers[(keys, a)].decision == "not" and db.sample_images == {}
    assert api.get(review_url(), headers=CAREGIVER).json()["count"] == 0


def test_yes_needs_a_valid_box(api, keys):
    a = api.app_state.db.photo(names=["keys"])
    assert answer(api, keys, a, "yes").status_code == 422
    assert answer(api, keys, a, "fix", bbox=[0.9, 0.9, 0.5, 0.5], width=640, height=480).status_code == 422


def test_answers_respect_the_photo_caps(api, keys, monkeypatch):
    db = api.app_state.db
    a = db.photo(names=["keys"])
    monkeypatch.setattr("app.routers.training.MAX_TRAINING_BYTES", 10)
    resp = answer(api, keys, a, "yes", bbox=[0.1, 0.1, 0.2, 0.2], width=640, height=480)
    assert resp.status_code == 409 and "full" in resp.json()["detail"]
    assert db.answers.get((keys, a)) is None  # not answered, so it stays in the queue

    monkeypatch.setattr("app.routers.training.MAX_TRAINING_BYTES", 10**9)
    db.training[keys][1].samples = [object()] * MAX_SAMPLES_PER_ITEM  # only counted
    resp = answer(api, keys, a, "yes", bbox=[0.1, 0.1, 0.2, 0.2], width=640, height=480)
    assert resp.status_code == 409 and str(MAX_SAMPLES_PER_ITEM) in resp.json()["detail"]


def test_answers_are_for_members_and_their_own_photos(api, keys):
    db = api.app_state.db
    theirs = db.photo(names=["keys"], device="someone-else@clients")
    assert answer(api, keys, theirs, "not").status_code == 404
    mine = db.photo(names=["keys"])
    assert api.post(review_url(OTHER), headers=CAREGIVER, json={"item_id": keys, "memory_id": mine, "decision": "not"}).status_code == 404
    assert answer(api, str(uuid.uuid4()), mine, "not").status_code == 404
