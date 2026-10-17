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
