# Security

LastScene handles photos and words from a person's day, so please report anything that could expose them privately, not in a public issue.

**Report a vulnerability:** use GitHub's [private vulnerability reporting](../../security/advisories/new) for this repository. Include what you found, how to reproduce it, and what it exposes. Please don't test against other people's households or data.

**What's in scope:** the API in `backend/` (including the read-only MCP server at `/mcp`), the dashboard in `web/`, and the wearable client in `device/`.

**For contributors:**

- Never commit secrets. Keys live in `backend/.env`, `device/device.env` and the hosting provider's environment settings; only the `.example` templates belong in git.
- Turn on the secret check once per clone: `git config core.hooksPath .githooks`.
- If a secret is ever committed, rotate it at its provider first (deleting the commit doesn't un-publish it), then remove it from history.

**Spending and quotas:** every model call (Gemini, and OpenAI or Claude when switched on) takes a slot from per-minute and per-day limits shared by all server instances through Postgres (`backend/app/limits.py`). Each wearable and each dashboard user also has its own limits. Defaults and how to tune them are in `backend/.env.example`. Over a limit the API answers 429 with `Retry-After`, and a provider's own 429/503 makes every instance back off. Paid features (Maps names, Live, Wi-Fi positioning) are off or capped by default.

**The MCP server (`/mcp`, see [docs/MCP.md](docs/MCP.md)):**

- It is read-only. No tool writes household data. Every request needs a household member's Auth0 token, checked like the dashboard's, or a household MCP key. Tools only reach households the caller belongs to, and a key reaches only its own household.
- MCP keys are random 256-bit values, shown once and stored as SHA-256 hashes. Only the household owner can make or revoke one, and a household can have at most 5.
- Each user gets 60 requests a minute, counted in the database (429 past that). Request bodies over 64 KiB get 413. Only POST is served.
- Tool calls are logged with the tool name, household, user and key id, never their arguments. Tools return no photos or photo links.
- Whatever a tool returns goes to the model that called it, Google included when it's Gemini. Treat an MCP key like a password, and revoke it on the Household page if it leaks.
