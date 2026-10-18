import numpy as np

from memory_engine.detector import Detection, detect_tiled, drop_oversized


def test_tiled_boxes_map_from_overlapping_crops_to_full_frame_coordinates():
    class LocalBoxDetector:
        def detect(self, image):
            return [Detection("keys", 0.5, (2, 3, 12, 13))]

    detections = detect_tiled(LocalBoxDetector(), np.zeros((100, 100, 3), dtype=np.uint8))

    assert [d.bbox for d in detections] == [
        (2, 3, 12, 13),       # full frame
        (2, 3, 12, 13),       # top-left tile (overlaps the full frame)
        (40, 3, 50, 13),      # top-right tile
        (2, 41, 12, 51),      # bottom-left tile
        (40, 41, 50, 51),     # bottom-right tile
    ]


def test_whole_frame_boxes_are_dropped():
    small = Detection("keys", 0.3, (100, 100, 150, 140))
    whole = Detection("keys", 0.08, (0, 0, 360, 640))  # the real false memory from desk_a.mov
    assert drop_oversized([small, whole], (640, 360, 3), 0.5) == [small]


def test_half_frame_is_the_limit():
    half = Detection("keys", 0.3, (0, 0, 100, 100))
    assert drop_oversized([half], (100, 200), 0.5) == [half]
    assert drop_oversized([half], (100, 199), 0.5) == []
