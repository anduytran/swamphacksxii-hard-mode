"""Shared test setup. Loaded before the test modules, so it applies to every Settings() they make."""

import os

# backend/.env may hold real raia keys; an empty environment variable wins over it, so no test can send a
# photo to the real agent. Tests that exercise raia pass a key and a mock transport themselves.
os.environ["RAIA_AGENT_KEY"] = ""
os.environ["RAIA_ORG_KEY"] = ""
