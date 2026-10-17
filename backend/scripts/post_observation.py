"""Try the vision-pipeline API from the command line (or copy it into the pipeline's uploader).

    # post a frame + one detection
    python scripts/post_observation.py post --image frame.jpg --label keys --scene "kitchen counter" \\
        --bbox 0.42 0.55 0.08 0.06 --confidence 0.71 --event placed
    # or post a full observation JSON the pipeline produced
    python scripts/post_observation.py post --image frame.jpg --json memory.json
    python scripts/post_observation.py last-seen keys          # latest_memory() / locate()
    python scripts/post_observation.py last-seen               # every item
    python scripts/post_observation.py found keys              # found()
    python scripts/post_observation.py delete <observation_id> # undo a wrong detection

Env: AUTH0_DOMAIN, AUTH0_AUDIENCE, DEVICE_CLIENT_ID, DEVICE_CLIENT_SECRET, PAIRING_CODE,
API_URL (default https://www.lastscene.photo). Cache the token as below in a real uploader:
Auth0's free plan caps machine-to-machine tokens per month.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

TOKEN_CACHE = Path.home() / ".lastscene-token.json"


def token() -> str:
    if TOKEN_CACHE.exists():
        cached = json.loads(TOKEN_CACHE.read_text())
        if cached.get("expires_at", 0) > time.time() + 60:
            return cached["access_token"]
    resp = httpx.post(
        f"https://{os.environ['AUTH0_DOMAIN']}/oauth/token",
        json={
            "grant_type": "client_credentials",
            "client_id": os.environ["DEVICE_CLIENT_ID"],
            "client_secret": os.environ["DEVICE_CLIENT_SECRET"],
            "audience": os.environ["AUTH0_AUDIENCE"],
        },
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    TOKEN_CACHE.write_text(json.dumps({"access_token": body["access_token"], "expires_at": time.time() + body.get("expires_in", 3600)}))
    TOKEN_CACHE.chmod(0o600)
    return body["access_token"]


def headers() -> dict:
    h = {"Authorization": f"Bearer {token()}"}
    if code := os.environ.get("PAIRING_CODE"):
        h["X-Pairing-Code"] = code
    return h


def show(resp: httpx.Response) -> None:
    print(resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    post = sub.add_parser("post")
    post.add_argument("--image", required=True)
    post.add_argument("--json", help="a full observation JSON file (otherwise built from the flags below)")
    post.add_argument("--label")
    post.add_argument("--scene")
    post.add_argument("--bbox", nargs=4, type=float, metavar=("X", "Y", "W", "H"))
    post.add_argument("--confidence", type=float, default=0.7)
    post.add_argument("--event", default="placed", choices=["placed", "seen", "moved", "picked_up"])
    post.add_argument("--size", nargs=2, type=int, default=[1280, 720], metavar=("W", "H"))
    ls = sub.add_parser("last-seen")
    ls.add_argument("label", nargs="?")
    fd = sub.add_parser("found")
    fd.add_argument("label")
    dl = sub.add_parser("delete")
    dl.add_argument("observation_id")
    args = p.parse_args()

    api = os.environ.get("API_URL", "https://www.lastscene.photo").rstrip("/")
    if args.cmd == "post":
        if args.json:
            observation = json.loads(Path(args.json).read_text())
        else:
            if not (args.label and args.bbox):
                sys.exit("--label and --bbox are required without --json")
            observation = {
                "observation_id": str(uuid.uuid4()),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "source": "auto",
                "pipeline": {"name": "post_observation.py", "version": "0"},
                "image_size": {"width": args.size[0], "height": args.size[1]},
                **({"scene": {"id": args.scene.replace(" ", "-"), "label": args.scene}} if args.scene else {}),
                "objects": [{"label": args.label, "confidence": args.confidence, "bbox": args.bbox, "event": args.event}],
            }
        image = Path(args.image)
        with image.open("rb") as f:
            files = {"image": (image.name, f, "image/png" if image.suffix.lower() == ".png" else "image/jpeg")}
            show(httpx.post(f"{api}/api/v1/observations", headers=headers(), files=files,
                            data={"observation": json.dumps(observation)}, timeout=60))
    elif args.cmd == "last-seen":
        path = f"/api/v1/items/{args.label}/last-seen" if args.label else "/api/v1/items/last-seen"
        show(httpx.get(f"{api}{path}", headers=headers(), timeout=30))
    elif args.cmd == "found":
        show(httpx.post(f"{api}/api/v1/items/{args.label}/found", headers=headers(), json={}, timeout=30))
    else:
        show(httpx.delete(f"{api}/api/v1/observations/{args.observation_id}", headers=headers(), timeout=30))


if __name__ == "__main__":
    main()
