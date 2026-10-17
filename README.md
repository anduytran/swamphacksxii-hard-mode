# swamphacksxii-hard-mode
LastScene: multimodal ai embedded on hardware to track down items

## Layout

```
api/index.py      Vercel entry point for the API (the dashboard is static, built from web/)
backend/          FastAPI service
  app/main.py         app + /config.json (+ serves web/dist locally)
  app/auth.py         Auth0 JWT validation (M2M device vs. caregiver user)
  app/db.py           TigerData / Postgres + pgvector: transcripts, embeddings and photos
  app/signing.py      short-lived signed photo links for the dashboard
  app/ai.py           every AI stage, routed per stage to Gemini (default), OpenAI or Claude
  app/gemini.py       Gemini: speech-to-text, embeddings, answers, extraction, vision, TTS
  app/routers/        memory, observations (vision pipeline), dashboard, households, public, images, admin (see API)
  app/retention.py    72h purge job (run from cron)
  scripts/device_sim.py  pretend-wearable client for end-to-end testing
web/              caregiver dashboard: React + TypeScript + Vite (UI spec: web/DESIGN.md)
  src/App.tsx         shell + hash routing (#/items …); map pages load as their own chunks
  src/pages/          Items, Timeline, Map, Insights, Search, Public map, Household, Landing
  src/components/     shell, charts (bar list, columns, sortable table), item drawer, forms
  src/lib/            Auth0 (auth.tsx), API client and types, data cache, deck.gl layers, public grid
  src/styles.css      design tokens + all styles
device/           Raspberry Pi Zero 2 W wearable client, wiring guide, systemd unit
deploy/           systemd unit + crontab for the Vultr instance
```

## API

Base URL `https://www.lastscene.photo` (the bare domain redirects there). Interactive docs at `/docs`, schema at `/openapi.json`.

Auth: **device** = Auth0 machine-to-machine token (`Authorization: Bearer …`, scope `write:memories`), optionally with an `X-Pairing-Code` header so a caregiver can claim the device. **caregiver** = the dashboard user's Auth0 token. **owner** / **member** = caregiver who owns / belongs to that household. Everything a device or caregiver sees is scoped to its household.

### Wearable (device)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/v1/memory/log` | multipart `audio`, `image` (JPEG/PNG); optional location `wifi` (JSON `[{bssid, signal}]`) or `lat`, `lng`, `accuracy_m` | `{id, timestamp, transcript}` |
| `POST /api/v1/memory/query` | multipart `audio` (+ same optional location fields) | spoken answer (`audio/wav`); `X-Question` / `X-Answer` headers hold the URL-encoded text |

### Vision pipeline (device)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/v1/observations` | multipart `image` (uncropped frame) + `observation` (JSON string, below) + optional `audio` | 201 `{memory_id, items, duplicate: false}` |
| `DELETE /api/v1/observations/{observation_id}` | – | `{deleted, items}`; each item's last-seen rolls back to its previous sighting |
| `GET /api/v1/items/last-seen` | – | `{items: [LastSeen, …]}` for the household |
| `GET /api/v1/items/{label}/last-seen` | – | one `LastSeen` (label is case-insensitive), 404 if never seen |
| `POST /api/v1/items/{label}/found` | optional `{found_at, observation_id}` | `{label, found_at}` |

`observation` JSON:

```json
{
  "observation_id": "6f1c…-uuid",
  "captured_at": "2026-10-17T14:03:12.481Z",
  "source": "auto",
  "pipeline": {"name": "lastscene-vision", "version": "0.3.0", "detector": "yolo-world", "matcher": "orb"},
  "image_size": {"width": 1280, "height": 720},
  "scene": {"id": "kitchen-01", "label": "kitchen counter", "match_score": 0.82},
  "objects": [
    {"label": "keys", "confidence": 0.71, "bbox": [0.42, 0.55, 0.08, 0.06], "track_id": "keys-3", "event": "placed"}
  ]
}
```

- `captured_at` must carry a timezone; `bbox` is normalised `[x, y, w, h]` (top-left origin) and must lie inside the frame; `event` is one of `placed`, `seen`, `moved`, `picked_up`.
- Status codes: 201 stored · 200 same `observation_id` again from the same device (`duplicate: true`, safe to retry) · 409 that id belongs to another device · 422 invalid JSON/fields, no object ≥ `MIN_DETECTION_CONFIDENCE`, or `captured_at` > 5 min ahead · 429 more than one observation per `OBSERVATION_MIN_INTERVAL_SECONDS` (has `Retry-After`).

`LastSeen`:

```json
{
  "label": "keys", "memory_id": "…", "observation_id": "…", "source": "auto",
  "captured_at": "2026-10-17T14:03:12.481000+00:00",
  "scene": {"id": "kitchen-01", "label": "kitchen counter"},
  "bbox": [0.42, 0.55, 0.08, 0.06], "confidence": 0.71, "maybe": false,
  "image_size": {"width": 1280, "height": 720},
  "image_url": "https://www.lastscene.photo/api/v1/images/ls-…?exp=…&sig=…",
  "found_at": null
}
```

### Dashboard (caregiver)

| Endpoint | Returns |
|---|---|
| `GET /api/v1/dashboard/history` | `[{id, device_id, timestamp, source, transcript, scene, summary, detections, image_url, items}]`, newest first |
| `GET /api/v1/dashboard/items` | `{window_hours, items}`: per item its last location, photo + `detection` box, earlier spots, mentions, times asked |
| `GET /api/v1/dashboard/insights?hours=` | counts, most-asked / most-logged items, unanswered questions, recent Q&A, activity; `hours` narrows the window (default and maximum 72) |
| `GET /api/v1/dashboard/missing-times?days=` | questions asked and things put down per hour (UTC instants) for the "When things go missing" grid; `days` 7–90, default 28. Older than 72 h these come from `activity_hourly`: anonymous per-device hourly counts the purge keeps for 90 days (no words, photos or item names) |
| `GET /api/v1/dashboard/summary` | `{summary}`: 2–3 sentences from Gemini (cached 10 min) |
| `GET /api/v1/dashboard/search?q=` | `{query, results}`: hybrid vector + keyword retrieval with per-chunk match explanations |
| `GET /api/v1/dashboard/map` | house floor plan: rooms, item spots, movement trips |
| `GET /api/v1/dashboard/places` | the household's places (private included) and located memories |

### Households (caregiver)

| Endpoint | Who | Body | Returns |
|---|---|---|---|
| `GET /api/v1/households/me` | caregiver | – | `{households}` with members, devices, places |
| `POST /api/v1/households` | caregiver | `{name, display_name?}` | `{id}` |
| `POST /api/v1/households/{id}/devices` | owner | `{client_id, pairing_code, label?}` | `{device_id, memories_placed}` |
| `POST /api/v1/households/{id}/invites` | owner | – | `{code, expires_in_hours}` (one use) |
| `POST /api/v1/invites/accept` | caregiver | `{code}` | `{household_id}` |
| `PATCH /api/v1/households/{id}` | owner | `{share_public}` | public-map opt-in |
| `PATCH /api/v1/households/{id}/places/{place_id}` | member | `{name?, kind?}` (`kind`: `private` or `public`) | `{places}` |

### Teach items (dashboard → vision pipeline)

| Endpoint | Who | Body | Returns |
|---|---|---|---|
| `GET /api/v1/households/{id}/training/items` | member | – | `{items, limits}`: each item's name, other words, photos (signed links) and boxes |
| `POST /api/v1/households/{id}/training/items` | member | `{name, aliases?}` | the item (409 if the name is taken; up to 50 items) |
| `PATCH /api/v1/households/{id}/training/items/{item_id}` | member | `{name?, aliases?}` | `{updated}` |
| `DELETE /api/v1/households/{id}/training/items/{item_id}` | member | – | `{deleted}` (its photos too) |
| `POST /api/v1/households/{id}/training/items/{item_id}/samples` | member | multipart `image` + `bbox` (`[x, y, w, h]`, 0..1) + `width` + `height` | the photo (up to 30 per item, 2 MB, JPEG/PNG/WebP) |
| `DELETE /api/v1/households/{id}/training/samples/{sample_id}` | member | – | `{deleted}` |
| `GET /api/v1/training/vocabulary` | device | – | `{version, classes, items}`: names for YOLO-World's `set_classes`; class id = position |
| `GET /api/v1/training/dataset` | device | – | `{version, names, samples}`: each photo's signed link, box in pixels and YOLO label line |
| `POST /api/v1/training/models` | device | `{version, files: [{name, size, sha256}], metrics}` | `{id, part_bytes, parts}`; then `PUT .../models/{id}/files/{name}/parts/{n}` (raw, ≤ 3 MB each) and `POST .../models/{id}/complete` (checks sha256, makes it current; the newest 2 are kept) |
| `GET /api/v1/training/models/current` | device | – | the newest model's manifest; download with `GET .../models/{id}/files/{name}/parts/{n}` |
| `GET /api/v1/households/{id}/training/model` | member | – | `{photos, model, up_to_date}` for the Teach items page |

### Public, photos and ops

| Endpoint | Auth | Returns |
|---|---|---|
| `GET /api/v1/public/lost-map` | none | anonymous ~100 m cells (k ≥ 3 households) |
| `GET /api/v1/images/{id}?exp=…&sig=…` | signed link from the endpoints above (15 min) | the photo (`ls-…` ids are last-seen frames) |
| `GET /api/v1/admin/purge` | `Authorization: Bearer $CRON_SECRET` | `{deleted, retention_hours, last_seen_deleted}` (Vercel Cron, daily 03:15 UTC) |
| `GET /healthz` | none | `{ok, missing_settings, wifi_positioning, geolocation_budget}`: whether the Geolocation key is set (never its value) and this month's calls against the cap |
| `GET /config.json` | none | public Auth0 values for the dashboard |

Queries only search the calling device's own memories from the last 72h (top 3 by cosine distance).

After each `log` and `query`, Gemini extracts the items mentioned ("keys" → "kitchen counter") into `memory_items`, and questions are recorded in `query_logs`. These feed the dashboard's **Items** and **Insights** pages and are purged with everything else after 72h.

## Setup

### Auth0
1. **API**: create one with identifier = `AUTH0_AUDIENCE` (e.g. `https://api.lastscene`), RS256. Add permission `write:memories`.
2. **Wearable**: create a *Machine to Machine* application, authorize it for the API with the `write:memories` scope. Put its Client ID/Secret on the wearable ([device/README.md](device/README.md)); the device POSTs `client_credentials` to `https://AUTH0_DOMAIN/oauth/token` and sends `Authorization: Bearer <access_token>` (cache it until `expires_in`).
3. **Dashboard**: create a *Single Page Application*. Allowed Callback / Logout / Web Origins = your dashboard URL (e.g. `http://localhost:8000`). Put its Client ID in `AUTH0_SPA_CLIENT_ID`.
4. Because any user who can log in sees the feed, disable public sign-ups on the database connection, or enable RBAC on the API, create a `read:history` permission for caregivers, and set `CAREGIVER_PERMISSION=read:history`.

### TigerData
Create a service and put its connection string in `DATABASE_URL`. On startup the app runs `CREATE EXTENSION vector` and creates the `memory_logs` table with an HNSW index. Photos are stored in the same table (`bytea`), so there is no separate object store; at 72h retention and ~100 KB per JPEG this stays tiny.

### Photo link secret
Set `IMAGE_URL_SECRET` to a long random string (`python3 -c "import secrets; print(secrets.token_urlsafe(32))"`). It signs the photo links the dashboard loads; changing it just invalidates links already handed out.

### Run locally

```bash
cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
cp backend/.env.example backend/.env
```

```bash
cd backend && .venv/bin/uvicorn app.main:app --reload
```

Build the dashboard once (Node 20+), and again after frontend changes; FastAPI serves `web/dist`:

```bash
cd web && npm install && npm run build
```

The dashboard is at http://localhost:8000. For live reload, run `cd web && npm run dev` (http://localhost:5173, proxying the API to :8000); signing in there needs `http://localhost:5173` added to the Auth0 Dashboard app's callback, logout and web-origin URLs. Frontend type check and tests:

```bash
cd web && npm run typecheck && npm test
```

Backend tests (no external services needed):

```bash
cd backend && .venv/bin/pytest -q
```

Simulate three days of the wearable without spending any tokens (no Gemini/Claude/OpenAI or Auth0 calls). It runs the real device client and API routes in-process with offline stand-ins for every AI stage, and writes to `DATABASE_URL`, so the dashboard shows the result. Rows use the device id `sim-wearable@clients` and are replaced on each run:

```bash
cd backend && .venv/bin/python scripts/simulate_hardware.py
```

```bash
cd backend && .venv/bin/python scripts/simulate_hardware.py --clean
```

The simulator no longer touches the real public map. `--public-demo` still seeds 3 synthetic opted-in households onto it, but for demos use the public map's **Simulated** toggle instead, which generates data in the browser and saves nothing.

The scenario covers saving items, a location filled in from the photo, a question about an item that was never logged, and a memory spooled while offline and uploaded later. Viewing the Insights page will still request the real Gemini summary.

Try it end to end as the wearable:

```bash
cd backend && DEVICE_CLIENT_ID=... DEVICE_CLIENT_SECRET=... AUTH0_DOMAIN=... AUTH0_AUDIENCE=... .venv/bin/python scripts/device_sim.py log --audio keys.wav --image counter.jpg
```

### Deploy on Vercel at lastscene.photo

The Vercel project uses the FastAPI preset with Root Directory `backend` (and "include files outside the root directory" on). Each deploy installs `requirements.txt`, then `vercel.json`'s build command builds the dashboard (`web/`, `npm ci && npm run build`); the FastAPI app serves the API and the built `web/dist`. `vercel.json` also holds the daily purge schedule. `.vercelignore` keeps `.env`, the venv, `node_modules` and the device code out of uploads.

1. Install the CLI and log in (opens a browser):
   ```bash
   npm i -g vercel
   ```
   ```bash
   vercel login
   ```
2. From the repo root, create the project (accept the defaults; `vercel.json` sets the build):
   ```bash
   vercel link
   ```
3. Add the environment variables for **Production** in the Vercel dashboard (Project → Settings → Environment Variables), copying values from `backend/.env`: `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_SPA_CLIENT_ID`, `DATABASE_URL`, `IMAGE_URL_SECRET`, `GEMINI_API_KEY`, plus:
   - `ENRICH_IN_BACKGROUND=false` (serverless functions may freeze once the response is sent, so enrichment runs before replying)
   - `CRON_SECRET=` a long random string (Vercel Cron sends it to `/api/v1/admin/purge`)
4. Deploy:
   ```bash
   vercel --prod
   ```
5. Point the domain: Project → Settings → Domains → add `lastscene.photo` (and `www.lastscene.photo`, redirecting to the apex). Vercel shows the DNS records to create at your registrar, normally an `A` record for `@` → `76.76.21.21` and a `CNAME` for `www` → `cname.vercel-dns.com`. HTTPS is issued automatically once DNS resolves.
6. In Auth0 → Applications → **Dashboard** (SPA) → Settings, make sure `https://lastscene.photo` is in **Allowed Callback URLs**, **Allowed Logout URLs** and **Allowed Web Origins**.
7. On the wearable, set `API_URL=https://lastscene.photo` in `/etc/lastscene/device.env`.

Limits to know: Vercel caps request bodies at 4.5 MB (a 15 s 16 kHz WAV is ~0.5 MB and the camera JPEG ~0.2 MB, so the wearable stays well under), and each function call is capped at `maxDuration` (60 s).

### Deploy on a VPS instead (Vultr)
Clone to `/opt/lastscene`, create the venv and `.env` as above, then:

```bash
sudo cp deploy/lastscene.service /etc/systemd/system/ && sudo systemctl enable --now lastscene
```

```bash
crontab deploy/lastscene.cron
```

Put it behind HTTPS (Caddy or nginx + Let's Encrypt). Auth0 requires HTTPS origins for the SPA outside localhost, and the device's token should never travel over plain HTTP.

## Households, places and the public map

LastScene is usually set up by a family member **for** the person who wears it.

1. **Household page → Set up for someone** creates a household; the creator is its owner.
2. **Add a wearable** with the device's Auth0 Client ID and the pairing code on its label (`PAIRING_CODE` in `device.env`). The device must have connected once; the first code it sends is the one that claims it (trust on first use). A wrong code and an unknown device get the same error.
3. **Invite someone** gives a one-use code (72 h) that family members enter under **Join a household**.

**Positions.** The Pi Zero 2 W has no GPS. With `LOCATION_SOURCE=wifi` it sends the access points it can hear with each memory and question; the server resolves them with Google's Geolocation API (`GOOGLE_GEOLOCATION_API_KEY`) and stores only the resulting coordinates, never the scan. `LOCATION_SOURCE=fixed` sends `HOME_LAT`/`HOME_LNG` instead (also used as a fallback).

**Staying inside Google's free tier.** Every Geolocation call is counted in the database *before* it is made, against `GEOLOCATION_MONTHLY_LIMIT` (default 5,000 per calendar month, UTC). The counter is atomic, so parallel serverless requests can't overshoot it. Once the month is used up, or if the database can't be checked, no call is made and memories fall back to the device's own position. Answers are cached in the database by access-point set for `GEOLOCATION_CACHE_DAYS` (30), so places the wearer visits often cost nothing after the first lookup. `/healthz` shows `geolocation_budget: {month, limit, used}`. For a second, Google-side limit, cap the Geolocation API's requests per day under *APIs & Services → Geolocation API → Quotas* in Google Cloud.

**Public place names (opt-in).** With `MAPS_GROUNDING=true`, a memory at a place its household marked public gets a name from Gemini's Grounding with Google Maps ("Starbucks, Reitz Union"), in the background after it is located. Home and other private places are never sent. Answers are cached in Postgres by position (~11 m) for `MAPS_CACHE_DAYS` (180), and at most `MAPS_DAILY_LIMIT` (50) calls are made a day, which stays under the 5,000 free requests a month. The Timeline, Map → Our places and the public map show the name followed by its linked Google Maps sources, as Google's terms require. Spoken answers keep the household's own place wording, since a speaker can't show the sources.

**Places** are built automatically: each position joins the household's nearest place (within its radius or the fix's accuracy) or starts a new one. The most-visited place is labelled Home until someone names places. Indoors, rooms still come from the spoken words and the photo (the **House** floor plan); GPS-level places drive **Map → Our places**.

**Privacy model.**

| Data | Who sees it |
|---|---|
| Memories, photos, questions, private places (homes, by default every place) | Members of the wearer's household only. Every dashboard endpoint is scoped to the caller's household devices. |
| Places a member marked **public** (shops, parks, transit) | Feed the anonymous public map only if the owner turned on **Share anonymous counts** (off by default), and only for memories from then on. |
| Public map (`/#/public`, no login) | Item names and counts per ~100 m cell, from the last 90 days, shown only where **3 or more households** contributed. No words, photos, device or household ids, or exact positions. |

## Vision pipeline (automatic observations)

The vision source is in [`vision/`](vision/) (see `AGENTS.md`). For the demo, the wearable streams frames
to a laptop that runs `vision/lastscene_bridge`; the [integration plan](docs/bridge-integration.md) covers
the API contract, pairing and live acceptance checks. `vision/scripts/live_smoke.py` checks a deployment
end to end once a device's credentials are set.

A separate pipeline (YOLO-World detection + ORB scene matching, on a host machine) creates memories automatically. It posts one **observation** per event (an item placed, moved, picked up, or seen again) to `POST /api/v1/observations`: the uncropped frame plus JSON with labels, confidences, normalised `[x, y, w, h]` boxes, scene id/label, track ids and `captured_at`. The full format, rules and open questions are in the integration contract doc; `app/observations.py` validates it.

- Stored as a memory with `source = 'auto'`, detections in `memory_detections`. Searchable from its structured labels (item and scene chunks, embedded like any memory; no Gemini caption), so voice answers and Search include it.
- Each item's newest `placed` / `seen` / `moved` sighting is kept in `item_last_seen` with its own copy of the frame and box, shared across the household, for `latest_memory()` / `locate()` / `guide()`. Late or replayed uploads never overwrite a newer sighting; deleting an observation rolls back to the previous one.
- Limits: detections below `MIN_DETECTION_CONFIDENCE` (0.35) are dropped, below `MAYBE_CONFIDENCE` (0.5) are flagged "maybe"; one observation per device every `OBSERVATION_MIN_INTERVAL_SECONDS` (2) or 429; `captured_at` more than 5 minutes ahead is rejected.
- Retention: observations follow the 72 h purge. `LAST_SEEN_RETENTION_HOURS` (default 72) controls the last-seen frames; set 720 if the team adopts the 30-day proposal.
- The dashboard draws the boxes over photos (dashed + "?" when maybe) with a "Seen automatically" badge.
- Try it: `backend/scripts/post_observation.py` (post, last-seen, found, delete) with a device's Auth0 credentials.
- **Training and models.** `vision/lastscene_pipeline`: `sync` → `train` → `publish` on a laptop, or `vultr run` on a temporary Vultr server with spending guardrails ([vision/docs/VULTR.md](vision/docs/VULTR.md)); wearables run `fetch` to get the newest model.
- **Teach items.** On the dashboard's **Teach items** page, people add their own items and photos of them, with a box drawn around the item in each. The pipeline reads the names from `GET /api/v1/training/vocabulary` (re-fetch when `version` changes) and the photos from `GET /api/v1/training/dataset`; `backend/scripts/fetch_training_set.py dataset` writes them as a YOLO folder (`data.yaml`, `images/`, `labels/`, `prompts.json` for visual prompts) to fine-tune on. These photos are kept until someone deletes them, not purged after 72 h.

Live directions and visibility checks run next to the camera (`lastscene_bridge.run --show`: w/f/q).
Backend last-seen is historical evidence. The found endpoint marks `found_at` and retains
server evidence; the bridge's local deletion is separate. Keep cluttered surfaces,
distant keys, room recognition and untested ESP32 capture out of the demo.

## Where the ML lives

Gemini runs every stage by default (`app/gemini.py`); each stage can be switched back to its original provider with an env var.

| Stage | Default (Gemini) | Alternative | Env var | Code |
|---|---|---|---|---|
| Speech → text | `gemini-3.8-flash` (audio in) | Whisper | `STT_PROVIDER` | `AIServices.transcribe` |
| Text → vector | `gemini-embedding-2` (1536-d) | OpenAI `text-embedding-3-small` | `EMBEDDING_PROVIDER` | `AIServices.embed` |
| Semantic search | pgvector HNSW, cosine distance | – | – | `MemoryDB.nearest` |
| Answer the question | `gemini-3.8-flash`, grounded on the top-3 memories | Claude | `LLM_PROVIDER` | `AIServices.answer` |
| Item + location extraction | `gemini-3.8-flash` structured output | Claude | `LLM_PROVIDER` | `AIServices.extract_items` |
| Photo understanding | `gemini-3.8-flash` vision: one-line scene + items | off | `VISION_PROVIDER` | `AIServices.describe_scene` |
| Caregiver summary | `gemini-3.8-flash` | Claude | `LLM_PROVIDER` | `AIServices.caregiver_summary` |
| Text → speech | `gemini-3.8-flash-tts` (24 kHz WAV) | OpenAI TTS | `TTS_PROVIDER` | `AIServices.speak` |

After each memory is saved, Gemini looks at the photo, fills in the location when the wearer didn't say one ("I'm putting my keys down" → "table by the front door"), and the memory is re-embedded with that description so questions match what the camera saw as well as what was said. Changing `EMBEDDING_PROVIDER` makes earlier memories unsearchable until they age out (72h).

The wearable runs no models by default. Next steps with an obvious home: on-device wake word / VAD (openWakeWord, Silero) in `device/lastscene_device.py`, and pattern detection for caregivers in `routers/dashboard.py::_insights`.

## Adding a visualization

Data flows `TigerData → routers/dashboard.py (insights) → web/src/pages/Insights.tsx → web/src/components/Charts.tsx`.

1. **Compute it on the server.** Add a field to the dict returned by `_insights()` in `backend/app/routers/dashboard.py`. Prefer raw timestamps (as `activity` does) when the chart groups by local time, so the browser can bucket in the viewer's timezone. Add a test in `backend/tests/test_api.py`, and the field to `Insights` in `web/src/lib/types.ts`.
2. **Draw it.** In `InsightsPage` (`web/src/pages/Insights.tsx`), add a `<section className="card">` with an `<h2>` at the marked spot, and a chart from the kit:
   - `<BarList rows={[{label, value, key}]} unit empty onSelect? />` for rankings and top-N (`onSelect` makes rows open the item drawer)
   - `<Columns categories series={[{name, values}]} label />` for time buckets or comparisons (1–2 series; add a `.legend`)
   - `bucketByHour(events, buckets)` (`lib/format.ts`) to count timestamped events into local-time buckets
   - `<DataTable headers rows />` inside `<details>` for the text view every chart should have
3. **Colors.** Use only `--series-1` / `--series-2` for data; status colors are reserved for good/warning badges. A third series needs a validated color first (see the comment at the top of `Charts.tsx`).

## Security and costs

- **Secrets live only in env files and Vercel.** `backend/.env`, `device/device.env` and printed `labels/` are git-ignored; only the `.example` templates are committed.
- **Turn on the secret check once per clone:** `git config core.hooksPath .githooks`. The pre-commit hook blocks secret files, the real values in your local `backend/.env` / `device/device.env`, and common key formats (Google, OpenAI, Anthropic, AWS, GitHub, Slack, private keys, database URLs with a password). It never prints a value.
- **Who can spend money:** every paid AI call (Gemini) and Google geolocation call is made for a paired wearable, which needs the household's own Auth0 device credentials. Anyone can sign up for the dashboard, so for an account without a wearable: search returns nothing without calling Gemini, Teach items is off, and one account can belong to at most 5 households. Teach-items photos are capped at 100 MB per household.
- **Google geolocation** stops at `GEOLOCATION_MONTHLY_LIMIT` (5000) calls a month, inside Google's free allowance; `/healthz` shows the month's count.
- **Google Maps place names** are off unless `MAPS_GROUNDING=true`, then capped at `MAPS_DAILY_LIMIT` (50) calls a day across the app and cached by position.

## Privacy
The daily purge (Vercel Cron → `/api/v1/admin/purge`, or `python -m app.retention` from cron on a VPS) deletes every `memory_logs` row older than `RETENTION_HOURS` (72), photo included. Photos are only reachable through signed links that expire after 15 minutes.
