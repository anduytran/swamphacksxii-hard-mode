# Handoff: Oct 18 (Sun, ~5:45 AM), before the 8:00 AM Devpost deadline

Read with [`AGENTS.md`](../AGENTS.md) (rules and decisions). This replaces the Oct 17 vision handoff: everything
below is on `master` and deployed at https://www.lastscene.photo.

## What runs

| Piece | Where | Started by |
|---|---|---|
| API + dashboard | Vercel, `www.lastscene.photo` (the apex redirects; devices must use `www`) | push to `master` |
| Wearable (audio Pi, `sreer@100.78.33.50`) | `lastscene-device` systemd service, runs as root | `device/demo_live.sh` |
| Camera Pi (`faker@100.89.232.47`) | `lastscene-camera` systemd service: YOLO11s INT8 passive detection + `/stream` on :8765 | `device/demo_live.sh` |
| Camera view | `device/camera_view.py` on the laptop, http://127.0.0.1:8766 (adds the camera token) | `device/demo_live.sh` |

`device/demo_live.sh` pulls both Pis, restarts their services, checks them, opens the camera view and follows the
wearable's log. Each Pi asks for its password once.

## Verified on Oct 18

- Tests: backend 362 passed, web 38, device 90, vision 243 (8 + 15 skipped need local clips or `RUN_DB_TESTS=1`;
  the DB integration tests pass with it).
- Live: wake commands (save, ask, found, "this is meeting room 2360"), passive sightings with RAIA context,
  Google Wi-Fi positioning at the Reitz Union (±19 m), distance and direction answers, the Live page with sightings.
- No secrets in tracked files or git history (only `.example` placeholders).

## Decisions made Oct 18

- Answers about an item always get its last 5 LastScene locations and where it is from the wearer; there are no
  follow-up commands ("continue" was removed). A newer spoken memory beats an older camera sighting.
- RAIA reads an item at 0.5 confidence or more, once per item per 30 min, silently.
- Rooms come from Wi-Fi fingerprints the household names by voice; the wearer's phone hotspot is ignored.
  A named place also saves its position, used when Google/BeaconDB can't place a scan.

## Before the demo

1. Run `device/demo_live.sh` (the camera Pi's server was down at 5:15 AM).
2. Walk the demo rooms with the wearable: "Last Scene, this is meeting room 2350" in each, near the middle.
3. Clean test data off the dashboard (the "Wrights" place, "geo test" items) so judges don't see it.

## Known limits (keep out of the demo)

- Cluttered surfaces and far-away items are detected poorly; rooms next to each other can share a fingerprint.
- RAIA's description sits on one sighting; while the camera keeps seeing an item, newer sightings push it out of
  the last 5 (the "LastScene:" memory stays, so answers still use it).
- The Pi needs Wi-Fi: no answers offline (memories spool and upload later).
