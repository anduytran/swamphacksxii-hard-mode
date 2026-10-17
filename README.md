# swamphacksxii-hard-mode
LastScene: multimodal ai embedded on hardware to track down items

## Layout

```
api/index.py      Vercel entry point (serves backend/ + frontend/)
backend/          FastAPI service
  app/main.py         app + static dashboard + /config.json
  app/auth.py         Auth0 JWT validation (M2M device vs. caregiver user)
  app/db.py           TigerData / Postgres + pgvector: transcripts, embeddings and photos
  app/signing.py      short-lived signed photo links for the dashboard
  app/ai.py           every AI stage, routed per stage to Gemini (default), OpenAI or Claude
  app/gemini.py       Gemini: speech-to-text, embeddings, answers, extraction, vision, TTS
  app/routers/        /api/v1/memory/{log,query}, /api/v1/dashboard/history, /api/v1/images/{id}
  app/retention.py    72h purge job (run from cron)
  scripts/device_sim.py  pretend-wearable client for end-to-end testing
frontend/         plain HTML/JS caregiver dashboard: Items, Timeline, Insights pages (auth0-spa-js)
  charts.js           reusable chart kit (bar list, grouped columns, data table)
device/           Raspberry Pi Zero 2 W wearable client, wiring guide, systemd unit
deploy/           systemd unit + crontab for the Vultr instance
```

## API

| Endpoint | Auth | Body | Returns |
|---|---|---|---|
| `POST /api/v1/memory/log` | device M2M token with `write:memories` | multipart `audio`, `image` (JPEG) | `{id, timestamp, transcript}` |
| `POST /api/v1/memory/query` | device M2M token with `write:memories` | multipart `audio` | spoken answer (`audio/wav` by default); `X-Question` / `X-Answer` headers hold the URL-encoded text |
| `GET /api/v1/dashboard/history` | caregiver user token | – | `[{id, device_id, timestamp, transcript, image_url}]`, last 24h, newest first |
| `GET /api/v1/dashboard/items` | caregiver user token | – | one entry per item: last location, photo, earlier spots, times asked (72h) |
| `GET /api/v1/dashboard/insights` | caregiver user token | – | counts, most-asked / most-logged items, questions about items never logged, activity, recent Q&A (72h) |
| `GET /api/v1/dashboard/summary` | caregiver user token | – | `{summary}`: 2–3 sentences written by Gemini from the insights (cached 10 min) |
| `GET /api/v1/admin/purge` | `Authorization: Bearer $CRON_SECRET` | – | runs the 72h purge (Vercel Cron, daily 03:15 UTC) |
| `GET /api/v1/dashboard/map` | caregiver | – | House floor plan: rooms, item spots, movement trips |
| `GET /api/v1/dashboard/places` | caregiver | – | The household's places (private included) and located memories |
| `GET /api/v1/dashboard/search?q=` | caregiver | – | Hybrid retrieval with match explanations (one embedding call) |
| `GET /api/v1/households/me` · `POST /api/v1/households` | caregiver | `{name}` | Households you belong to, with members, devices and places · create one |
| `POST /api/v1/households/{id}/devices` | owner | `{client_id, pairing_code, label}` | Claim a wearable; places its earlier memories |
| `POST /api/v1/households/{id}/invites` · `POST /api/v1/invites/accept` | owner · caregiver | – · `{code}` | One-use family invite |
| `PATCH /api/v1/households/{id}` · `.../places/{place}` | owner · member | `{share_public}` · `{name, kind}` | Public-map opt-in · name a place, mark it public/private |
| `GET /api/v1/public/lost-map` | none | – | Anonymous public cells (k ≥ 3 households) |
| `POST /api/v1/observations` | device token | multipart `image` + `observation` JSON (+ optional `audio`) | Automatic memory from the vision pipeline: 201 `{memory_id, items}`; same `observation_id` again → 200 |
| `DELETE /api/v1/observations/{id}` | device token | – | Undo a wrong detection; last-seen rolls back |
| `GET /api/v1/items/last-seen` · `/items/{label}/last-seen` | device token | – | Last-seen frame (signed link), box, scene, time per item in the household |
| `POST /api/v1/items/{label}/found` | device token | `{found_at?}` | Mark an item found |
| `GET /api/v1/images/{id}?exp=…&sig=…` | signed link from the dashboard endpoints (15 min) | – | the photo |

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

The dashboard is at http://localhost:8000. Tests (no external services needed):

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

The scenario covers saving items, a location filled in from the photo, a question about an item that was never logged, and a memory spooled while offline and uploaded later. Viewing the Insights page will still request the real Gemini summary.

Try it end to end as the wearable:

```bash
cd backend && DEVICE_CLIENT_ID=... DEVICE_CLIENT_SECRET=... AUTH0_DOMAIN=... AUTH0_AUDIENCE=... .venv/bin/python scripts/device_sim.py log --audio keys.wav --image counter.jpg
```

### Deploy on Vercel at lastscene.photo

The whole app (API + dashboard) runs as one Python function: `api/index.py` imports the FastAPI app, `vercel.json` routes every path to it and schedules the daily purge. `.vercelignore` keeps `.env`, the venv and the device code out of uploads.

1. Install the CLI and log in (opens a browser):
   ```bash
   npm i -g vercel
   ```
   ```bash
   vercel login
   ```
2. From the repo root, create the project (accept the defaults; no build command):
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

**Places** are built automatically: each position joins the household's nearest place (within its radius or the fix's accuracy) or starts a new one. The most-visited place is labelled Home until someone names places. Indoors, rooms still come from the spoken words and the photo (the **House** floor plan); GPS-level places drive **Map → Our places**.

**Privacy model.**

| Data | Who sees it |
|---|---|
| Memories, photos, questions, private places (homes, by default every place) | Members of the wearer's household only. Every dashboard endpoint is scoped to the caller's household devices. |
| Places a member marked **public** (shops, parks, transit) | Feed the anonymous public map only if the owner turned on **Share anonymous counts** (off by default), and only for memories from then on. |
| Public map (`/#/public`, no login) | Item names and counts per ~100 m cell, from the last 90 days, shown only where **3 or more households** contributed. No words, photos, device or household ids, or exact positions. |

## Vision pipeline (automatic observations)

A separate pipeline (YOLO-World detection + ORB scene matching, on a host machine) creates memories automatically. It posts one **observation** per event (an item placed, moved, picked up, or seen again) to `POST /api/v1/observations`: the uncropped frame plus JSON with labels, confidences, normalised `[x, y, w, h]` boxes, scene id/label, track ids and `captured_at`. The full format, rules and open questions are in the integration contract doc; `app/observations.py` validates it.

- Stored as a memory with `source = 'auto'`, detections in `memory_detections`. Searchable from its structured labels (item and scene chunks, embedded like any memory; no Gemini caption), so voice answers and Search include it.
- Each item's newest `placed` / `seen` / `moved` sighting is kept in `item_last_seen` with its own copy of the frame and box, shared across the household, for `latest_memory()` / `locate()` / `guide()`. Late or replayed uploads never overwrite a newer sighting; deleting an observation rolls back to the previous one.
- Limits: detections below `MIN_DETECTION_CONFIDENCE` (0.35) are dropped, below `MAYBE_CONFIDENCE` (0.5) are flagged "maybe"; one observation per device every `OBSERVATION_MIN_INTERVAL_SECONDS` (2) or 429; `captured_at` more than 5 minutes ahead is rejected.
- Retention: observations follow the 72 h purge. `LAST_SEEN_RETENTION_HOURS` (default 72) controls the last-seen frames; set 720 if the team adopts the 30-day proposal.
- The dashboard draws the boxes over photos (dashed + "?" when maybe) with a "Seen automatically" badge.
- Try it: `backend/scripts/post_observation.py` (post, last-seen, found, delete) with a device's Auth0 credentials.

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

Data flows `TigerData → routers/dashboard.py (insights) → frontend/app.js (loadInsights) → frontend/charts.js`.

1. **Compute it on the server.** Add a field to the dict returned by `insights()` in `backend/app/routers/dashboard.py`. Prefer raw timestamps (as `activity` does) when the chart groups by local time, so the browser can bucket in the viewer's timezone. Add a test in `backend/tests/test_api.py`.
2. **Give it a place on the page.** In `frontend/index.html`, copy a `<section class="card">` with an `<h2>` and an empty `<div id="chart-…">` at the marked spot on the Insights page (add a `.legend` if it has two series).
3. **Draw it.** In `loadInsights()` call the chart kit:
   - `Charts.barList(el, [{label, value}], {unit, empty})` for rankings and top-N
   - `Charts.columns(el, {categories, series: [{name, values}]}, {label, table})` for time buckets or comparisons (1–2 series)
   - `Charts.bucketByHour(events, {buckets})` to count timestamped events into local-time buckets
   - `Charts.dataTable(table, headers, rows)` for the text view every chart should have
4. **Colors.** Use only `--series-1` / `--series-2` for data; status colors are reserved for good/warning badges. A third series needs a validated color first (see the comment at the top of `charts.js`).

## Privacy
The daily purge (Vercel Cron → `/api/v1/admin/purge`, or `python -m app.retention` from cron on a VPS) deletes every `memory_logs` row older than `RETENTION_HOURS` (72), photo included. Photos are only reachable through signed links that expire after 15 minutes.
