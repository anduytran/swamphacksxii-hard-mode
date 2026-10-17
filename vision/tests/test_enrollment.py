import numpy as np
import pytest


class Encoder:
    version = {"sha256": "test-model", "preprocessing": "test-v1"}

    def encode(self, crop):
        return np.array([1., 0.])


def test_persistence_and_delete(tmp_path):
    from memory_engine.enrollment import EnrollmentStore
    store = EnrollmentStore(tmp_path, Encoder())
    item = store.register("Adam's keys", "bunch of keys")
    store.add_example(item, np.zeros((20, 30, 3), np.uint8), (1, 2, 15, 18),
                      source="desk_b.mov", frame_ms=1000, confirmed=True)
    restored = EnrollmentStore(tmp_path, Encoder())
    assert restored.items()[0]["enrollment_version"] == 1
    assert restored.items()[0]["examples"][0]["frame_ms"] == 1000
    assert restored.match([1, 0], "bunch of keys", .8, .1)["item_id"] == item
    restored.delete(item)
    assert restored.items() == []
    assert not list(tmp_path.rglob("*.png"))


@pytest.mark.parametrize("box", [(-1, 0, 5, 5), (0, 0, 0, 4), (0, 0, 31, 10)])
def test_invalid_crops(tmp_path, box):
    from memory_engine.enrollment import EnrollmentStore
    store = EnrollmentStore(tmp_path, Encoder())
    item = store.register("keys", "bunch of keys")
    with pytest.raises(ValueError):
        store.add_example(item, np.zeros((20, 30, 3), np.uint8), box,
                          source="desk_b.mov", frame_ms=0, confirmed=True)


def test_confirmation_unknown_and_version(tmp_path):
    from memory_engine.enrollment import EnrollmentStore
    store = EnrollmentStore(tmp_path, Encoder())
    item = store.register("keys", "bunch of keys")
    kwargs = dict(source="desk_b.mov", frame_ms=0)
    with pytest.raises(ValueError):
        store.add_example(item, np.zeros((20, 30, 3), np.uint8), (0, 0, 10, 10), **kwargs)
    store.add_example(item, np.zeros((20, 30, 3), np.uint8), (0, 0, 10, 10), confirmed=True, **kwargs)
    assert store.match([0, 1], "bunch of keys", .8, .1)["status"] == "unknown"
    other = Encoder()
    other.version = {"sha256": "other", "preprocessing": "test-v1"}
    with pytest.raises(ValueError, match="version"):
        EnrollmentStore(tmp_path, other).match([1, 0], "bunch of keys", .8, .1)


def test_ambiguous_items_and_detector_confidence(tmp_path):
    from memory_engine.enrollment import EnrollmentStore, EnrolledDetector
    from memory_engine.detector import Detection
    store = EnrollmentStore(tmp_path, Encoder())
    for name in ["Adam keys", "Other keys"]:
        item = store.register(name, "bunch of keys")
        store.add_example(item, np.zeros((20, 30, 3), np.uint8), (0, 0, 10, 10),
                          source="test", frame_ms=0, confirmed=True)
    assert store.match([1, 0], "bunch of keys", .8, .1)["status"] == "ambiguous"
    store.delete(item)
    class Detector:
        def detect(self, image):
            return [Detection("bunch of keys", .054, (0, 0, 10, 10))]
    detector = EnrolledDetector(Detector(), store, .8, .1)
    result = detector.detect(np.zeros((20, 30, 3), np.uint8))[0]
    assert result.label == "adam keys"
    assert result.conf == .054
    assert result.bbox == (0, 0, 10, 10)


def test_named_label_ingestion_lookup_and_found(tmp_path):
    from memory_engine.enrollment import EnrollmentStore, EnrolledDetector
    from memory_engine.detector import Detection
    from memory_engine.api import LastSeen
    from lastscene_bridge.bridge import LastSceneBridge
    from lastscene_bridge.auth import Auth0ClientCredentials
    from lastscene_bridge.uploader import Uploader
    from tests import mock_lastscene as mock
    import cv2
    from pathlib import Path
    store = EnrollmentStore(tmp_path / "refs", Encoder())
    item = store.register("Adam's keys", "bunch of keys")
    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "bus.jpg"))
    box = (223, 406, 344, 860)
    store.add_example(item, image, box, source="synthetic-test", frame_ms=0, confirmed=True)
    class Detector:
        def detect(self, image): return [Detection("bunch of keys", .148, box)]
    det = EnrolledDetector(Detector(), store, .8, .1)
    ls = LastSeen(out_dir=tmp_path / "mem", detector=det, search_detector=det, scene_matching=False)
    srv, state, url = mock.serve(0)
    try:
        up = Uploader(url, mock.PAIRING_CODE,
                      Auth0ClientCredentials(url, mock.CLIENT_ID, mock.CLIENT_SECRET, mock.AUDIENCE), tmp_path / "queue")
        bridge = LastSceneBridge(ls, uploader=up, preset_name="m960d")
        for ms in [0, 500, 1000]:
            bridge.handle(ls.process_frame(image, ms, observed_ms=1792220000000+ms))
        bridge.handle(ls.finish())
        assert [status for _, status in up.flush()] == ["sent"]
        assert up.last_seen("Adam's keys").json()["label"] == "adam's keys"
        assert ls.latest_memory("adam's keys") is not None
        assert bridge.found("Adam's keys")["backend_status"] == 200
        assert "adam's keys" in state.found
        assert ls.latest_memory("adam's keys") is None
    finally:
        srv.shutdown()
        ls.close()
