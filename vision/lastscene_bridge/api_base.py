"""The API base URL, without the heavy upload dependencies (cv2, numpy), so fetch/publish stay light."""

import logging
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger("lastscene_bridge.uploader")


def normalize_api_base(api_base: str) -> str:
    """https://lastscene.photo 308-redirects to www, and requests drops Authorization on that hop (-> 401)."""
    parts = urlsplit(api_base.rstrip("/"))
    if parts.hostname == "lastscene.photo":
        fixed = urlunsplit(parts._replace(netloc="www.lastscene.photo" + (f":{parts.port}" if parts.port else "")))
        log.warning("API base %s redirects to www (bearer token would be dropped); using %s", api_base, fixed)
        return fixed
    return parts.geturl()
