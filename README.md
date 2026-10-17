# swamphacksxii-hard-mode
LastScene: multimodal ai embedded on hardware to track down items

## Layout

```
backend/          FastAPI service (Vultr VPS)
  app/main.py         app + static dashboard + /config.json
  app/auth.py         Auth0 JWT validation (M2M device vs. caregiver user)
  app/db.py           TigerData / Postgres + pgvector: transcripts, embeddings and photos
  app/signing.py      short-lived signed photo links for the dashboard
  app/ai.py           Whisper STT, OpenAI embeddings + TTS, Claude answer generation
  app/routers/        /api/v1/memory/{log,query}, /api/v1/dashboard/history, /api/v1/images/{id}
  app/retention.py    72h purge job (run from cron)
  scripts/device_sim.py  pretend-wearable client for end-to-end testing
frontend/         plain HTML/JS caregiver dashboard (auth0-spa-js)
deploy/           systemd unit + crontab for the Vultr instance
```

## API

| Endpoint | Auth | Body | Returns |
|---|---|---|---|
| `POST /api/v1/memory/log` | device M2M token with `write:memories` | multipart `audio`, `image` (JPEG) | `{id, timestamp, transcript}` |
| `POST /api/v1/memory/query` | device M2M token with `write:memories` | multipart `audio` | spoken answer (`audio/wav` by default); `X-Question` / `X-Answer` headers hold the URL-encoded text |
| `GET /api/v1/dashboard/history` | caregiver user token | – | `[{id, device_id, timestamp, transcript, image_url}]`, last 24h, newest first |
| `GET /api/v1/images/{id}?exp=…&sig=…` | signed link from `/history` (15 min) | – | the photo |

Queries only search the calling device's own memories from the last 72h (top 3 by cosine distance).

## Setup

### Auth0
1. **API**: create one with identifier = `AUTH0_AUDIENCE` (e.g. `https://api.lastscene`), RS256. Add permission `write:memories`.
2. **Wearable**: create a *Machine to Machine* application, authorize it for the API with the `write:memories` scope. Flash its Client ID/Secret onto the ESP32; the device POSTs `client_credentials` to `https://AUTH0_DOMAIN/oauth/token` and sends `Authorization: Bearer <access_token>` (cache it until `expires_in`).
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

## Privacy
`python -m app.retention` (daily via cron, 03:15) deletes every `memory_logs` row older than `RETENTION_HOURS` (72), photo included. Photos are only reachable through signed links that expire after 15 minutes.
