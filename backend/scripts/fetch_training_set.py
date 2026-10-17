"""Download the items a household taught on the dashboard, as a YOLO dataset (for the pipeline host).

    python scripts/fetch_training_set.py vocabulary            # names for YOLO-World's set_classes
    python scripts/fetch_training_set.py dataset --out lastscene-items

`dataset` writes a standard YOLO folder:

    lastscene-items/
      data.yaml                   names + paths (train and val both point at the photos)
      images/train/<id>.jpg       the photos people took
      labels/train/<id>.txt       one line each: "class cx cy w h", 0..1
      prompts.json                per photo: class + box in pixels, for visual prompts (YOLOE)
      version                     re-run only when GET /training/vocabulary's version changes

Fine-tune with Ultralytics, e.g. `yolo detect train data=lastscene-items/data.yaml model=yolov8s-worldv2.pt
epochs=50 imgsz=640`, or skip training and call `model.set_classes(vocabulary)`. With a few photos per
item, fine-tuning mostly helps tell your own keys from other keys; check results on frames from the
wearable before switching models.

Env as in post_observation.py: AUTH0_DOMAIN, AUTH0_AUDIENCE, DEVICE_CLIENT_ID, DEVICE_CLIENT_SECRET,
PAIRING_CODE, API_URL (default https://www.lastscene.photo).
"""

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from post_observation import headers  # noqa: E402  (token caching lives there)

API = os.environ.get("API_URL", "https://www.lastscene.photo").rstrip("/")


def get(path: str) -> dict:
    resp = httpx.get(f"{API}{path}", headers=headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def write_dataset(data: dict, out: Path, http: httpx.Client | None = None) -> int:
    """Write the YOLO folder for a GET /training/dataset response; returns the photo count."""
    images, labels = out / "images" / "train", out / "labels" / "train"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    keep = set()
    prompts = []
    with http or httpx.Client(timeout=30) as http:
        for s in data["samples"]:
            resp = http.get(s["image_url"])  # signed link: no token needed, but it expires
            resp.raise_for_status()
            ext = mimetypes.guess_extension(resp.headers.get("content-type", "image/jpeg").split(";")[0]) or ".jpg"
            name = f"{s['id']}{'.jpg' if ext == '.jpe' else ext}"
            (images / name).write_bytes(resp.content)
            (labels / f"{s['id']}.txt").write_text(s["yolo"] + "\n")
            keep |= {name, f"{s['id']}.txt"}
            prompts.append({"image": f"images/train/{name}", "class_id": s["class_id"], "name": s["name"], "bbox_xyxy": s["bbox_xyxy_px"]})
    for folder in (images, labels):  # photos deleted on the dashboard go here too
        for f in folder.iterdir():
            if f.name not in keep:
                f.unlink()
    names = "\n".join(f"  {k}: {json.dumps(v)}" for k, v in sorted(data["names"].items(), key=lambda kv: int(kv[0])))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/train\nnames:\n{names}\n")
    (out / "prompts.json").write_text(json.dumps(prompts, indent=2))
    (out / "version").write_text(data["version"])
    return len(data["samples"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("vocabulary")
    ds = sub.add_parser("dataset")
    ds.add_argument("--out", type=Path, default=Path("lastscene-items"))
    args = p.parse_args()
    if args.cmd == "vocabulary":
        print(json.dumps(get("/api/v1/training/vocabulary"), indent=2))
    else:
        count = write_dataset(get("/api/v1/training/dataset"), args.out)
        print(f"{count} photos -> {args.out}/ (data.yaml, images/, labels/, prompts.json)")


if __name__ == "__main__":
    main()
