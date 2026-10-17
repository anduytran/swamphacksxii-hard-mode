import numpy as np


def test_letterbox_and_distractor_suppression():
    from memory_engine.onnx_detector import prepare, decode
    image = np.zeros((100, 200, 3), np.uint8)
    tensor, transform = prepare(image, 320)
    assert tensor.shape == (1, 3, 320, 320)
    # Same box, keyboard beats keys. NMS must happen before filtering distractors.
    output = np.array([[[160, 160], [160, 160], [64, 64], [64, 64],
                        [.6, .1], [.1, .9], [.1, .1]]], dtype=np.float32)
    assert decode(output, transform, image.shape, .05) == []
    output = output[:, :, :1]
    result = decode(output, transform, image.shape, .05)[0]
    assert result.bbox == (80, 30, 120, 70)
    assert abs(result.conf - .6) < 1e-6


def test_threshold_failure_and_freeze():
    from scripts.evaluate_enrollment import choose_threshold
    import pytest
    threshold = choose_threshold([.9, .95], [.3, .6])
    assert .6 < threshold < .9
    with pytest.raises(ValueError):
        choose_threshold([.4], [.7])
