"""Upload a trained model to the LastScene API, so wearables can download it (fetch).

    python -m lastscene_pipeline.publish [--model DIR] [--env-file FILE]

Sends models/current (or --model): model-int8.onnx plus model.json, classes.json and metrics.json, in
parts of up to 3 MB (Vercel caps a request at about 4.5 MB). The server checks every file's sha256 before
the model becomes the household's current one. Exit 3 when the server already has this exact model.
"""

import argparse
import hashlib
from pathlib import Path

from lastscene_pipeline import EXIT_CONFIG, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline.common import Api, PipelineError, data_dir, load_env_file, read_json, resolve_model_dir, run_main

MODEL_FILES = ("model-int8.onnx", "model.json", "classes.json", "metrics.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_publish(model: Path, api: Api | None = None, log=print) -> int:
    folder = resolve_model_dir(model)
    if folder is None or not (folder / "model-int8.onnx").is_file():
        raise PipelineError(EXIT_CONFIG, f"no model-int8.onnx in {model} (run train first)")
    version = folder.name
    files = {name: (folder / name).read_bytes() for name in MODEL_FILES if (folder / name).is_file()}
    manifest = [{"name": n, "size": len(b), "sha256": sha256(b)} for n, b in files.items()]
    metrics = read_json(folder / "metrics.json") if "metrics.json" in files else {}
    api = api or Api()

    current = api.request("GET", "/api/v1/training/models/current", ok=(200, 404))
    if current.status_code == 200:
        have = {f["name"]: f["sha256"] for f in current.json()["files"]}
        if current.json()["version"] == version and have == {f["name"]: f["sha256"] for f in manifest}:
            log(f"the server already has model {version}")
            return EXIT_NOTHING

    created = api.request("POST", "/api/v1/training/models", ok=(201,),
                          json={"version": version, "files": manifest,
                                "metrics": {k: metrics.get(k) for k in ("map50", "base_model", "epochs", "imgsz", "trained_at") if k in metrics}}).json()
    part = created["part_bytes"]
    for name, data in files.items():
        for i in range(created["parts"][name]):
            api.request("PUT", f"/api/v1/training/models/{created['id']}/files/{name}/parts/{i}", ok=(204,),
                        data=data[i * part:(i + 1) * part], headers={"Content-Type": "application/octet-stream"})
    api.request("POST", f"/api/v1/training/models/{created['id']}/complete", ok=(200,))
    log(f"published model {version} ({sum(len(b) for b in files.values()) / 1e6:.1f} MB)")
    return EXIT_OK


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, help="models/<version> or models/current (default $LASTSCENE_DATA/models/current)")
    p.add_argument("--env-file", help="optional KEY=VALUE credentials file (never commit it)")
    args = p.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    return run_publish(args.model or data_dir() / "models" / "current")


if __name__ == "__main__":
    run_main(main)
