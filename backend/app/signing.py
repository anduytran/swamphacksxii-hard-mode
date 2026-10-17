"""Short-lived signed links for photos, so <img> tags can load them without a bearer token.

A link is /api/v1/images/{id}?exp=<unix time>&sig=<HMAC-SHA256 of "id:exp">.
"""

import hashlib
import hmac
import time


def _sign(secret: str, record_id: str, exp: int) -> str:
    if not secret:
        raise RuntimeError("IMAGE_URL_SECRET is not set")
    return hmac.new(secret.encode(), f"{record_id}:{exp}".encode(), hashlib.sha256).hexdigest()


def image_url(secret: str, record_id: str, ttl_seconds: int) -> str:
    exp = int(time.time()) + ttl_seconds
    return f"/api/v1/images/{record_id}?exp={exp}&sig={_sign(secret, record_id, exp)}"


def verify(secret: str, record_id: str, exp: int, sig: str) -> bool:
    if exp < time.time():
        return False
    return hmac.compare_digest(_sign(secret, record_id, exp), sig)
