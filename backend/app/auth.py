"""Auth0 JWT validation for the two kinds of callers.

* Wearable: Auth0 Machine-to-Machine (client credentials) access token.
* Caregiver: access token issued to the dashboard SPA via Universal Login.

Both are RS256 JWTs for the same API audience, verified against the tenant JWKS.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.concurrency import run_in_threadpool

from .config import Settings, get_settings

bearer = HTTPBearer(auto_error=False)


@dataclass
class Principal:
    sub: str
    claims: dict
    scopes: set[str] = field(default_factory=set)
    permissions: set[str] = field(default_factory=set)

    @property
    def is_machine(self) -> bool:
        return self.claims.get("gty") == "client-credentials" or self.sub.endswith("@clients")


@lru_cache
def _jwks_client(domain: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"https://{domain}/.well-known/jwks.json", cache_keys=True)


def decode_token(token: str, settings: Settings) -> dict:
    if not settings.auth0_domain or not settings.auth0_audience:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Auth0 is not configured")
    try:
        signing_key = _jwks_client(settings.auth0_domain).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.auth0_audience,
            issuer=f"https://{settings.auth0_domain}/",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc


async def get_principal(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    # PyJWKClient may fetch the JWKS over the network, so keep it off the event loop.
    claims = await run_in_threadpool(decode_token, creds.credentials, settings)
    return Principal(
        sub=claims["sub"],
        claims=claims,
        scopes=set(claims.get("scope", "").split()),
        permissions=set(claims.get("permissions", [])),
    )


async def require_device(
    principal: Principal = Depends(get_principal),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if not principal.is_machine:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Device token required")
    if settings.device_scope and settings.device_scope not in principal.scopes:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing scope {settings.device_scope}")
    return principal


async def require_caregiver(
    principal: Principal = Depends(get_principal),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if principal.is_machine:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "User token required")
    needed = settings.caregiver_permission
    if needed and needed not in principal.permissions | principal.scopes:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing permission {needed}")
    return principal
