# LastScene MCP server

`https://www.lastscene.photo/mcp` is a read-only [Model Context Protocol](https://modelcontextprotocol.io) server. Gemini, or any MCP client, can use it to look up a household's data and answer a caregiver's question, such as "When did Mom last have her wallet?".

- **Code:** `backend/app/mcp_server.py` (server and tools), `backend/app/db_mcp.py` (keys and the rate counter), `backend/app/routers/mcp_keys.py` (key endpoints).
- **Tests:** `backend/tests/test_mcp.py`.
- **SDK:** the official MCP Python SDK (`mcp==2.2.0`, `MCPServer`).

## What it exposes

Seven tools. All are read-only (`readOnlyHint: true`) and return small JSON, as `structuredContent` and as text. No tool returns photos or photo links.

| Tool | Arguments | Returns |
|---|---|---|
| `list_households` | – | the caller's households: `id`, `name`, `role`, number of wearables. Call it first. |
| `search_memories` | `household_id`, `query` (≤ 300 chars), `hours` = 72, `limit` = 5 (≤ 10) | memories that match, best first. Uses the same hybrid vector + keyword search as the wearable (`pipeline.retrieve`), and each result carries its text chunks. |
| `item_last_seen` | `household_id`, `item` (partial names match) | `memories`: the newest memories that name the item, with where it was put. `camera`: the vision pipeline's last sightings (`item_last_seen`), with confidence, `maybe` and `found_at`. |
| `timeline` | `household_id`, `since`, `until` (ISO 8601, UTC if no offset), `limit` = 20 (≤ 100) | memories (what was said, the scene, items) and the wearer's questions with their answers, newest first, plus `more`. |
| `items` | `household_id` | the Items page data: per item its last spot, times mentioned and asked, and earlier spots. Also `taught`, the items the household taught the camera. |
| `missing_times` | `household_id`, `days` = 28 (7–90) | the "When things go missing" grid (`/dashboard/missing-times`): per UTC hour, questions `asked` and things `put_down`. Counts only. |
| `places` | `household_id` | the household's places: label, name, private/public, home, centre, radius, visits. |

The tools reuse the dashboard's code (`dashboard.items`, `dashboard.missing_times`, `households._places_view`) and the existing `MemoryDB` methods. They add no SQL of their own. Each tool's description and JSON schema, plus the server's `instructions`, are what the model sees. Memories are kept for 72 hours (`RETENTION_HOURS`), so older questions find nothing, except in `missing_times`.

## Auth

Send `Authorization: Bearer <token>` on every request. Two kinds of token work:

1. **A caregiver's Auth0 access token.** This is the same token the dashboard uses, checked by the same code (`get_principal` + `require_caregiver`: signature, audience, issuer, expiry, `CAREGIVER_PERMISSION`). Device (M2M) tokens get 403. Tools see only households the caregiver belongs to.
2. **A household MCP key** (`lsk_…`), for scripts and headless use such as a Gemini call.
   - The household owner makes and revokes keys on the **Household** page (MCP keys card), or through the API below.
   - A key reads only its own household.
   - The key is shown once. Only its SHA-256 is stored.
   - A key stops working when it is revoked, or when the member who made it leaves the household.
   - A household can have up to 5 keys.

| Endpoint | Who | Returns |
|---|---|---|
| `GET /api/v1/households/{id}/mcp-keys` | member | `{keys: [{id, label, hint, created_at, last_used_at}]}`, never the key itself |
| `POST /api/v1/households/{id}/mcp-keys` `{label}` | owner | `{id, key, note}`: the only time the key is shown (409 past 5 keys) |
| `DELETE /api/v1/households/{id}/mcp-keys/{key_id}` | owner | `{revoked}` |

Errors:

- No token, or a bad, expired or revoked one: HTTP **401**, with `WWW-Authenticate: Bearer`.
- A `household_id` the caller can't see: the tool result has `isError: true` and the text `403 Forbidden: you are not a member of that household…`. It's in the tool result because JSON-RPC answers tool calls inside an HTTP 200. Unknown ids, malformed ids and other households' ids all get this same answer, so ids can't be probed.

## Limits

- **Rate:** 60 MCP requests a minute per caregiver. A key counts against the caregiver who made it. Every request counts: `initialize`, `tools/list` and each tool call. Over the limit you get HTTP **429** with `Retry-After: 60`. The counter is kept in the database (`mcp_rate`), so all serverless instances share it.
- **Size:** request bodies over 64 KiB get HTTP **413**, before they are parsed.
- **Transport:** **Streamable HTTP only** (POST, JSON responses). The server is stateless: every request stands alone and there are no session ids, which suits Vercel. There is no SSE transport and no GET stream. GET and DELETE on `/mcp` get 405. Server-to-client requests (sampling, elicitation) aren't used.
- Use **`www.lastscene.photo`**. The apex domain answers with a 308 redirect, and some clients drop the `Authorization` header on that redirect.

## Privacy

This is personal data about someone with memory loss: what they said, where they were, what they lost.

- **Read-only.** No tool writes, updates or deletes anything. The only writes an MCP request causes are its rate-counter row and a key's `last_used_at`.
- **Logging.** Each tool call is logged with the tool name, household id, caregiver id and key id. Its arguments are never logged: search queries and item names can be personal. Refused calls are logged as `denied`.
- **Where the data goes.** Anything a tool returns goes to whichever model or client called it. With Gemini, that means Google: the data, and the bearer token in `headers`, go to Google for that call. The example script below passes `store=False` so Google doesn't keep the interaction. Prefer a household MCP key over a personal Auth0 token: a key is narrower, and it can be revoked without signing out.
- **Photos.** No tool returns photos or photo links.

## Gemini example

Gemini calls remote MCP servers itself through the Interactions API. You give it the URL and the headers. Server names must be snake_case, with no `-`.

```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    system_instruction="You help a family caregiver. Answer from the LastScene tools; call list_households first.",
    input="When did Mom last have her wallet?",
    tools=[{
        "type": "mcp_server",
        "name": "lastscene",
        "url": "https://www.lastscene.photo/mcp",
        "headers": {"Authorization": f"Bearer {LASTSCENE_MCP_TOKEN}"},
        # google-genai 2.31 takes [{mode, tools}] here
        "allowed_tools": [{"mode": "auto", "tools": ["list_households", "item_last_seen", "search_memories", "timeline"]}],
    }],
    store=False,
)
print(interaction.output_text)
```

`backend/scripts/ask_gemini_mcp.py` is a runnable version of this example. It reads `GEMINI_API_KEY` and `LASTSCENE_MCP_TOKEN`, and optionally `LASTSCENE_MCP_URL`, from `backend/.env` or `--env-file`, and it never prints the token.

```bash
cd backend && .venv/bin/python scripts/ask_gemini_mcp.py --list-tools
```

```bash
cd backend && .venv/bin/python scripts/ask_gemini_mcp.py "When did Mom last have her wallet?"
```

`--list-tools` checks the token and URL against `/mcp` directly and makes no Gemini call.

## curl example

Put the token in a shell variable, not in a file you commit:

```bash
read -rs TOKEN
```

Initialize:

```bash
curl -s https://www.lastscene.photo/mcp -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
```

List the tools. Stateless, so no session id is needed:

```bash
curl -s https://www.lastscene.photo/mcp -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -H "MCP-Protocol-Version: 2025-06-18" -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

Call a tool:

```bash
curl -s https://www.lastscene.photo/mcp -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -H "MCP-Protocol-Version: 2025-06-18" -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_households","arguments":{}}}'
```

## Running the tests

```bash
cd backend && .venv/bin/python -m pytest -q tests/test_mcp.py
```

The tests drive the server with the SDK's own Streamable HTTP client, in process against the ASGI app. They cover both the `initialize` handshake and the 2026 discover path, plus plain JSON-RPC posts, every tool on fake data, 401, 403 for other households, 429, 413, the logs and the key lifecycle. `tests/test_mcp_integration.py` checks the key and rate-counter SQL against a real database. It is opt-in: `RUN_DB_TESTS=1`.
