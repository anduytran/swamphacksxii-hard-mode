# swamphacksxii-hard-mode
LastScene: multimodal ai embedded on hardware to track down items

## Layout

```
backend/          FastAPI service (Vultr VPS)
  app/main.py         app + static dashboard + /config.json
  app/auth.py         Auth0 JWT validation (M2M device vs. caregiver user)
  app/db.py           TigerData / Postgres + pgvector: transcripts, embeddings and photos
  app/signing.py      short-lived signed photo links for the dashboard
  app/ai.py           Whisper STT, OpenAI embeddings + TTS, Claude answers + item extraction
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
| `GET /api/v1/images/{id}?exp=…&sig=…` | signed link from the dashboard endpoints (15 min) | – | the photo |

Queries only search the calling device's own memories from the last 72h (top 3 by cosine distance).

After each `log` and `query` response is sent, Claude extracts the items mentioned ("keys" → "kitchen counter") into `memory_items`, and questions are recorded in `query_logs`. These feed the dashboard's **Items** and **Insights** pages and are purged with everything else after 72h.

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

Try it end to end as the wearable:

```bash
cd backend && DEVICE_CLIENT_ID=... DEVICE_CLIENT_SECRET=... AUTH0_DOMAIN=... AUTH0_AUDIENCE=... .venv/bin/python scripts/device_sim.py log --audio keys.wav --image counter.jpg
```

### Deploy on Vultr
Clone to `/opt/lastscene`, create the venv and `.env` as above, then:

```bash
sudo cp deploy/lastscene.service /etc/systemd/system/ && sudo systemctl enable --now lastscene
```

```bash
crontab deploy/lastscene.cron
```

Put it behind HTTPS (Caddy or nginx + Let's Encrypt). Auth0 requires HTTPS origins for the SPA outside localhost, and the device's token should never travel over plain HTTP.

## Where the ML lives

| Stage | Model | Where | Code |
|---|---|---|---|
| Speech → text | Whisper (`whisper-1`) | cloud | `AIServices.transcribe` |
| Text → vector | OpenAI `text-embedding-3-small` (1536-d) | cloud | `AIServices.embed` |
| Semantic search | pgvector HNSW, cosine distance | TigerData | `MemoryDB.nearest` |
| Answer the question | Claude, grounded on the top-3 memories | cloud | `AIServices.answer` |
| Item + location extraction | Claude structured outputs | cloud, background | `AIServices.extract_items` |
| Text → speech | `gpt-4o-mini-tts` (WAV) | cloud | `AIServices.speak` |

The wearable runs no models by default. Good next additions, each with an obvious home:

- **On-device wake word / VAD** (openWakeWord, Silero VAD) in `device/lastscene_device.py`, replacing the buttons and trimming silence before upload.
- **Photo understanding**: send the JPEG to Claude (vision) in `_index_items` to name the item and place even when the wearer doesn't say them, and embed that caption next to the transcript so search uses both.
- **Pattern detection** for caregivers (items that keep moving, questions that spike at night) computed in `routers/dashboard.py::insights` and drawn on the Insights page.

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
`python -m app.retention` (daily via cron, 03:15) deletes every `memory_logs` row older than `RETENTION_HOURS` (72), photo included. Photos are only reachable through signed links that expire after 15 minutes.
