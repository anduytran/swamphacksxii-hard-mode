"""Send one photo to the raia agent the way the backend does (app/raia.py) and print every step.

    .venv/bin/python scripts/raia_context_test.py --image frame.jpg --label pin          # like a wake photo
    .venv/bin/python scripts/raia_context_test.py --image frame.jpg --label keys --bbox 0.4 0.5 0.1 0.1  # observation

Reads RAIA_AGENT_KEY (and RAIA_API_URL) from backend/.env or the environment. Uses its own conversation user
(fkId lastscene-<--scope>, "smoke-test" by default), so test conversations stay apart from real households.
Prints the JSON context and the "LastScene: " memory the backend would save. Nothing is written to the database.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.raia import PREFIX, Raia, chunks_for, parse_context, place_of, prompt  # noqa: E402

TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


async def show(resp: httpx.Response) -> None:
    req = resp.request
    where = req.url.path if req.url.host != "api.raia2.com" else req.url.path.removeprefix("/external")
    host = "" if req.url.host == "api.raia2.com" else f" ({req.url.host})"
    print(f"  {req.method:5} {where}{host} -> {resp.status_code}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path, help="a JPEG/PNG/WebP photo with the item in it")
    ap.add_argument("--label", default="keys", help="the item's name, as the detector would label it")
    ap.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("X", "Y", "W", "H"),
                    help="a detector box, normalised [x y w h] from the top left, as observations send it "
                         "(default: none, as wake photos are sent)")
    ap.add_argument("--scene", default=None, help="optional camera location, as detect --scene-label sends it")
    ap.add_argument("--scope", default="smoke-test", help="conversation user fkId suffix (default smoke-test)")
    args = ap.parse_args()

    settings = Settings()
    if not settings.raia_agent_key:
        print("RAIA_AGENT_KEY is empty: put it in backend/.env (or the environment).", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"No such image: {args.image}", file=sys.stderr)
        return 2
    image_type = TYPES.get(args.image.suffix.lower(), "image/jpeg")
    image = args.image.read_bytes()

    print(f"raia at {settings.raia_api_url}, agent key set, org key {'set' if settings.raia_org_key else 'empty'}")
    print(f"Photo {args.image} ({len(image) / 1024:.0f} KB, {image_type}), item {args.label!r}, box {args.bbox}")
    print(f"Prompt: {prompt(args.label, args.bbox, args.scene)}\n")

    http = httpx.AsyncClient(timeout=httpx.Timeout(settings.raia_timeout_seconds, connect=10.0),
                             event_hooks={"response": [show]})
    raia = Raia(settings, http=http)
    started = datetime.now(timezone.utc)
    try:
        reply, conversation_id = await raia.describe(args.scope, args.label, args.bbox, args.scene, image, image_type,
                                                     started)
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        await http.aclose()

    print(f"\nAgent replied in {seconds:.1f} s:\n  {reply!r}")
    ctx = parse_context(reply)
    if not ctx:
        print("The backend would save nothing (not visible, or an empty reply).")
        return 0
    print("JSON context (saved in raia_contexts):\n  " + json.dumps(ctx, indent=2).replace("\n", "\n  "))
    print(f"Saved as a memory with transcript:\n  {PREFIX}{ctx['summary']}")
    print(f"  item: {args.label} -> {place_of(ctx)}")
    for kind, text in chunks_for(args.label, ctx):
        print(f"  chunk [{kind}] {text}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
