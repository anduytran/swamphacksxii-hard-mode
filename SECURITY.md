# Security

LastScene handles photos and words from a person's day, so please report anything that could expose them privately, not in a public issue.

**Report a vulnerability:** use GitHub's [private vulnerability reporting](../../security/advisories/new) for this repository. Include what you found, how to reproduce it, and what it exposes. Please don't test against other people's households or data.

**What's in scope:** the API in `backend/`, the dashboard in `web/`, and the wearable client in `device/`.

**For contributors:**

- Never commit secrets. Keys live in `backend/.env`, `device/device.env` and the hosting provider's environment settings; only the `.example` templates belong in git.
- Turn on the secret check once per clone: `git config core.hooksPath .githooks`.
- If a secret is ever committed, rotate it at its provider first (deleting the commit doesn't un-publish it), then remove it from history.
