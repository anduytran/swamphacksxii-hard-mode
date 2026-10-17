"""Use a real tiny ONNX graph to check the low-memory runtime configuration."""
import json

import numpy as np
import pytest

from lastscene_pipeline.runtime import OnnxYoloDetector, sha256


def fixed_pin_model(tmp_path):
    """A one-class "pin" model whose output is always one box at the centre of the 320 px input, score 0.6."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    raw = np.array([[[160], [160], [50], [50], [.6]]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(raw))],
        "fixed_detection",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "model-int8.onnx"
    onnx.save(model, path)
    (tmp_path / "model.json").write_text(json.dumps({
        "base_model": "test", "version": "1", "classes": ["pin"], "imgsz": 320,
        "files": {path.name: {"sha256": sha256(path)}},
    }))
    return path


def test_low_memory_runtime_preserves_detection_and_sleeps_idle_workers(tmp_path):
    path = fixed_pin_model(tmp_path)
    normal = OnnxYoloDetector(path, threads=2)
    low = OnnxYoloDetector(path, threads=2, low_memory=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    expected = normal.detect(frame)
    assert len(expected) == 1
    assert expected[0].label == "pin"
    assert low.detect(frame) == expected
    options = low.session.get_session_options()
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert not options.enable_cpu_mem_arena
    assert not options.enable_mem_pattern


def test_pi_eval_tiled_mode_maps_every_tile_box_to_source_pixels(tmp_path):
    from scripts.eval_pi_detection import infer_tiled

    detector = OnnxYoloDetector(fixed_pin_model(tmp_path), threads=1, low_memory=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections, raw_max = infer_tiled(detector, frame, 0.35, grid=2, overlap=0.25)

    # 2x2 grid, 25% overlap on a 640x480 frame: 400x300 tiles at x 0/240, y 0/180
    (local,) = detector.detect(np.zeros((300, 400, 3), dtype=np.uint8))
    x1, y1, x2, y2 = local.bbox
    assert [d.bbox for d in detections] == [
        detector.detect(frame)[0].bbox,
        (x1, y1, x2, y2), (x1 + 240, y1, x2 + 240, y2),
        (x1, y1 + 180, x2, y2 + 180), (x1 + 240, y1 + 180, x2 + 240, y2 + 180),
    ]
    assert raw_max == pytest.approx(0.6)
