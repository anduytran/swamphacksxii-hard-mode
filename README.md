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
