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
    """A link valid for at least ttl_seconds. The expiry is rounded up to a step of a third of the
    TTL, so lists fetched again within a step give the same links and the browser reuses its
    cached photos instead of asking for every one again."""
    exp = int(time.time()) + ttl_seconds
    if ttl_seconds > 0:
        step = max(1, ttl_seconds // 3)
        exp = -(-exp // step) * step
    return f"/api/v1/images/{record_id}?exp={exp}&sig={_sign(secret, record_id, exp)}"


def verify(secret: str, record_id: str, exp: int, sig: str) -> bool:
    if exp < time.time():
        return False
    return hmac.compare_digest(_sign(secret, record_id, exp), sig)
