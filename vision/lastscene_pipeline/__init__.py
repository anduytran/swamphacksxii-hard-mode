"""The three pipeline scripts from Dylan's "Vision pipeline: required script spec" (Oct 17).

    python -m lastscene_pipeline.sync     pull Teach items -> YOLO dataset ($LASTSCENE_DATA/dataset)
    python -m lastscene_pipeline.train    fine-tune -> models/<version>/ (+ INT8 ONNX for the Pi), move models/current
    python -m lastscene_pipeline.detect   frames -> memories -> one observation per event

Exit codes (all scripts): 0 done, 1 bad input or config, 2 network or API error (safe to retry), 3 nothing to do.
sync/train need the laptop environment (requirements.txt); detect also runs on a Raspberry Pi with only
onnxruntime + OpenCV + numpy + requests (see pi/README.md).
"""

EXIT_OK, EXIT_CONFIG, EXIT_NETWORK, EXIT_NOTHING = 0, 1, 2, 3
