from memory_engine.detector import Detection, drop_oversized


def test_whole_frame_boxes_are_dropped():
    small = Detection("keys", 0.3, (100, 100, 150, 140))
    whole = Detection("keys", 0.08, (0, 0, 360, 640))  # the real false memory from desk_a.mov
    assert drop_oversized([small, whole], (640, 360, 3), 0.5) == [small]


def test_half_frame_is_the_limit():
    half = Detection("keys", 0.3, (0, 0, 100, 100))
    assert drop_oversized([half], (100, 200), 0.5) == [half]
    assert drop_oversized([half], (100, 199), 0.5) == []
