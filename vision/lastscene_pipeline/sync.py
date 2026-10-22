"""sync: pull what the household taught on the Teach items page and write it as a YOLO dataset.

    python -m lastscene_pipeline.sync [--out DIR] [--force] [--env-file FILE] [--skip NAME ...]

Same folder layout as backend/scripts/fetch_training_set.py, plus classes.json and the unchanged-version skip:

    data.yaml                    path, train/val: images/train, names by class id
    images/train/<sample id>.jpg each photo as uploaded
    labels/train/<sample id>.txt "class cx cy w h", 0..1 (the API's yolo field)
    prompts.json                 per photo: image, class_id, name, bbox_xyxy (pixels)
    images/negatives/            item-specific absent photos, excluded from YOLO training
    negatives.json              absent photo paths and their item class ids
    classes.json                 the vocabulary response as received
    version                      the dataset's version (written last: a failed sync never looks complete)

--skip leaves an item's photos out of training (case-insensitive name). The item keeps its class id, so
class ids still match the vocabulary, and the model just won't learn it.

Exit codes: 0 synced, 1 config, 2 network/API error, 3 version unchanged (no --force).
"""

import argparse
import json
import mimetypes
from pathlib import Path

import requests

from lastscene_pipeline import EXIT_NETWORK, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline.common import Api, PipelineError, data_dir, load_env_file, run_main


def current_version(out: Path) -> str | None:
    f = out / "version"
    return f.read_text(encoding="utf-8").strip() if f.is_file() else None


def write_dataset(vocab: dict, data: dict, out: Path, http=None) -> int:
    """Download every photo (signed links expire after ~15 min: fetch right away), write the folder.
    Returns the photo count."""
    images, labels = out / "images" / "train", out / "labels" / "train"
    negatives_dir = out / "images" / "negatives"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    negatives_dir.mkdir(parents=True, exist_ok=True)
    (out / "version").unlink(missing_ok=True)  # until everything below is written
    http = http or requests.Session()
    keep_images, keep_labels, keep_negatives = set(), set(), set()
    prompts, negatives = [], []
    for s in data["samples"]:
        try:
            resp = http.get(s["image_url"], timeout=30)  # signed link: no token
        except requests.RequestException as exc:
            raise PipelineError(EXIT_NETWORK, f"photo {s['id']}: {type(exc).__name__}") from None
        if resp.status_code != 200:
            raise PipelineError(EXIT_NETWORK, f"photo {s['id']}: HTTP {resp.status_code} (signed link expired? re-run sync)")
        ext = mimetypes.guess_extension(resp.headers.get("content-type", "image/jpeg").split(";")[0]) or ".jpg"
        name = f"{s['id']}{'.jpg' if ext in ('.jpe', '.jpeg') else ext}"
        if s["yolo"]:
            (images / name).write_bytes(resp.content)
            (labels / f"{s['id']}.txt").write_text(s["yolo"] + "\n", encoding="utf-8")
            keep_images.add(name)
            keep_labels.add(f"{s['id']}.txt")
            prompts.append({"image": f"images/train/{name}", "class_id": s["class_id"], "name": s["name"],
                            "bbox_xyxy": s["bbox_xyxy_px"]})
        else:
            # Absent is scoped to one item; an empty YOLO label would mark every class absent.
            (negatives_dir / name).write_bytes(resp.content)
            keep_negatives.add(name)
            negatives.append({"image": f"images/negatives/{name}", "class_id": s["class_id"], "name": s["name"]})
    for folder, keep in ((images, keep_images), (labels, keep_labels), (negatives_dir, keep_negatives)):
        for f in folder.iterdir():
            if f.name not in keep:
                f.unlink()
    # class id = position in the vocabulary's classes, the same in every file (items without photos too)
    names = {str(i): n for i, n in enumerate(vocab["classes"])}
    if data.get("names") and data["names"] != names:
        raise PipelineError(EXIT_NETWORK, "vocabulary and dataset disagree on class ids (changed between requests); re-run sync")
    lines = "\n".join(f"  {k}: {json.dumps(v)}" for k, v in names.items())
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/train\n"
                                   f"names:\n{lines}\n", encoding="utf-8")
    (out / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    (out / "negatives.json").write_text(json.dumps(negatives, indent=2), encoding="utf-8")
    (out / "classes.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (out / "version").write_text(data["version"], encoding="utf-8")
    return len(data["samples"])


def skipped(name: str, skip) -> bool:
    return name.casefold() in {n.casefold() for n in skip}


def run_sync(out: Path, force: bool = False, api: Api | None = None, log=print, skip=()) -> int:
    api = api or Api()
    vocab = api.get_json("/api/v1/training/vocabulary")
    if not force and current_version(out) == vocab["version"]:
        log(f"dataset {vocab['version']} unchanged in {out}; nothing to do")
        return EXIT_NOTHING
    data = api.get_json("/api/v1/training/dataset")
    if data["version"] != vocab["version"]:  # someone edited between the two requests
        vocab = api.get_json("/api/v1/training/vocabulary")
    if skip:
        left_out = sum(skipped(s["name"], skip) for s in data["samples"])
        data = dict(data, samples=[s for s in data["samples"] if not skipped(s["name"], skip)])
        log(f"left out {left_out} photo(s) of {', '.join(skip)}")
    count = write_dataset(vocab, data, out)
    few = [f"{i['name']} ({i['samples']})" for i in vocab["items"] if i["samples"] < 5 and not skipped(i["name"], skip)]
    log(f"synced dataset {data['version']}: {len(vocab['classes'])} items, {count} photos -> {out}")
    if few:
        log(f"  fewer than 5 photos: {', '.join(few)}")
    return EXIT_OK


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, help="default $LASTSCENE_DATA/dataset")
    p.add_argument("--force", action="store_true", help="download even when the version is unchanged")
    p.add_argument("--env-file", help="optional KEY=VALUE credentials file (never commit it)")
    p.add_argument("--skip", action="append", default=[], help="item name whose photos stay out of training (repeatable)")
    args = p.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    return run_sync(args.out or data_dir() / "dataset", args.force, skip=args.skip)


if __name__ == "__main__":
    run_main(main)
