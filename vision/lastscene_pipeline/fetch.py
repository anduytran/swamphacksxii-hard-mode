"""Download the household's current model from the LastScene API (on the Pi; no torch needed).

    python -m lastscene_pipeline.fetch [--models DIR] [--env-file FILE]

Writes models/<version>/ with every file checked against its sha256, then points models/current at it,
so a running detect reloads it. Exit 3 when the newest model is already here or none is published yet.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

from lastscene_pipeline import EXIT_NETWORK, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline.common import Api, PipelineError, data_dir, load_env_file, resolve_model_dir, run_main, set_current


def run_fetch(models: Path, api: Api | None = None, log=print) -> int:
    api = api or Api()
    r = api.request("GET", "/api/v1/training/models/current", ok=(200, 404))
    if r.status_code == 404:
        log("no model published yet")
        return EXIT_NOTHING
    manifest = r.json()
    version = manifest["version"]
    here = resolve_model_dir(models / "current")
    if here is not None and here.name == version and all(
        (here / f["name"]).is_file() and hashlib.sha256((here / f["name"]).read_bytes()).hexdigest() == f["sha256"]
        for f in manifest["files"]
    ):
        log(f"model {version} is already current")
        return EXIT_NOTHING

    tmp = models / f".{version}.download"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in manifest["files"]:
        data = b"".join(
            api.request("GET", f"/api/v1/training/models/{manifest['id']}/files/{f['name']}/parts/{i}").content
            for i in range(f["parts"])
        )
        if len(data) != f["size"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
            shutil.rmtree(tmp, ignore_errors=True)
            raise PipelineError(EXIT_NETWORK, f"{f['name']} arrived incomplete or corrupted; run fetch again")
        (tmp / f["name"]).write_bytes(data)
    final = models / version
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    set_current(models, version)
    log(f"fetched model {version}")
    return EXIT_OK


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, help="default $LASTSCENE_DATA/models")
    p.add_argument("--env-file", help="optional KEY=VALUE credentials file (never commit it)")
    args = p.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    return run_fetch(args.models or data_dir() / "models")


if __name__ == "__main__":
    run_main(main)
