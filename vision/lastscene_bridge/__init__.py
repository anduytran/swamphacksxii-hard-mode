"""Bridge from the LastSeen ML pipeline to the LastScene backend (contract of Oct 17, 2026).

memory events -> selector (which events are worth sending) -> observation (contract JSON +
full-frame JPEG) -> uploader (Auth0 token, X-Pairing-Code, rate limit, retries, offline queue).
"Found it" is a queue item too, so no queued sighting from before it can reopen the item afterwards.
memory_engine is used through its public API only; nothing in it was changed for this.
"""

BRIDGE_VERSION = "0.2.0"
PIPELINE_NAME = "lastseen-ml"
