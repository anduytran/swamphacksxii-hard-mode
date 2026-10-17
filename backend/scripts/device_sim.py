"""Pretend to be the wearable: get an Auth0 M2M token, then log or query a memory.

    python scripts/device_sim.py log   --audio keys.wav --image counter.jpg
    python scripts/device_sim.py query --audio where.wav --out answer.wav

Env: AUTH0_DOMAIN, AUTH0_AUDIENCE, DEVICE_CLIENT_ID, DEVICE_CLIENT_SECRET, API_URL
"""

import argparse
import os
import sys
from urllib.parse import unquote

import httpx


def get_token() -> str:
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
    return resp.json()["access_token"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["log", "query"])
    p.add_argument("--audio", required=True)
    p.add_argument("--image")
    p.add_argument("--out", default="answer.wav")
    args = p.parse_args()

    api = os.environ.get("API_URL", "http://localhost:8000")
    headers = {"Authorization": f"Bearer {get_token()}"}
    files = {"audio": (os.path.basename(args.audio), open(args.audio, "rb"), "audio/wav")}

    if args.mode == "log":
        if not args.image:
            sys.exit("--image is required for log")
        files["image"] = (os.path.basename(args.image), open(args.image, "rb"), "image/jpeg")
        resp = httpx.post(f"{api}/api/v1/memory/log", headers=headers, files=files, timeout=60)
        resp.raise_for_status()
        print(resp.json())
    else:
        resp = httpx.post(f"{api}/api/v1/memory/query", headers=headers, files=files, timeout=60)
        resp.raise_for_status()
        with open(args.out, "wb") as f:
            f.write(resp.content)
        print("Q:", unquote(resp.headers.get("x-question", "")))
        print("A:", unquote(resp.headers.get("x-answer", "")))
        print("Saved audio to", args.out)


if __name__ == "__main__":
    main()
