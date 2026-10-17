"""Credentials for the LastScene API, from environment variables only (never from files in the repo).

Environment:
    LASTSCENE_API_BASE / API_URL  defaults to https://www.lastscene.photo
    LASTSCENE_PAIRING_CODE    the device's pairing code (sent as X-Pairing-Code)
  and either an Auth0 machine-to-machine client (contract):
    AUTH0_DOMAIN              e.g. your-tenant.us.auth0.com  (a full http(s):// URL also works, e.g. a local mock)
    AUTH0_CLIENT_ID
    AUTH0_CLIENT_SECRET
    AUTH0_AUDIENCE            the API identifier Dylan configured
  or a ready-made token (development only):
    LASTSCENE_DEVICE_TOKEN
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
import time

import requests


def _restrict_token_file(path):
    """0600 on POSIX; remove inherited Windows ACLs and grant current user only."""
    os.chmod(path, 0o600)
    if os.name == "nt":
        import csv
        import subprocess
        identity = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"],
                                  capture_output=True, text=True, check=True)
        sid = next(csv.reader([identity.stdout.strip()]))[1]
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:(F)"],
                       capture_output=True, check=True)


class StaticToken:
    def __init__(self, token: str):
        self._token = token

    def get(self) -> str:
        return self._token

    def invalidate(self):
        pass

    def __repr__(self):
        return "StaticToken(<redacted>)"


class Auth0ClientCredentials:
    """Auth0 client-credentials flow; caches the token until 60 s before it expires."""

    def __init__(self, domain: str, client_id: str, client_secret: str, audience: str, timeout: float = 10, cache_path=None):
        base = domain if domain.startswith(("http://", "https://")) else f"https://{domain}"
        self.url = base.rstrip("/") + "/oauth/token"
        self._body = {"grant_type": "client_credentials", "client_id": client_id,
                      "client_secret": client_secret, "audience": audience}
        self.timeout = timeout
        self._token, self._expires = None, 0.0
        self.cache_path = Path(cache_path).expanduser().resolve() if cache_path else None
        self.cache_key = hashlib.sha256(json.dumps([self.url, client_id, audience]).encode()).hexdigest()
        if self.cache_path:
            repo = Path(__file__).resolve().parents[2]
            if self.cache_path.is_relative_to(repo):
                raise ValueError("token cache must be outside the repository")
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if cached.get("key") == self.cache_key and float(cached["expires"]) > time.time():
                    self._token, self._expires = cached["token"], float(cached["expires"])
            except (OSError, ValueError, KeyError, TypeError):
                pass

    def get(self) -> str:
        if self._token is None or time.time() >= self._expires:
            r = requests.post(self.url, json=self._body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            self._token = data["access_token"]
            self._expires = time.time() + max(0, int(data.get("expires_in", 3600)) - 60)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(dir=self.cache_path.parent, prefix=".lastscene-token-")
                try:
                    _restrict_token_file(name)
                    with os.fdopen(fd, "w", encoding="utf-8") as stream:
                        json.dump(dict(key=self.cache_key, token=self._token, expires=self._expires), stream)
                    os.replace(name, self.cache_path)
                finally:
                    Path(name).unlink(missing_ok=True)
        return self._token

    def invalidate(self):
        self._token = None
        if self.cache_path:
            self.cache_path.unlink(missing_ok=True)

    def __repr__(self):
        return f"Auth0ClientCredentials(url={self.url!r}, client_secret=<redacted>)"


def token_provider_from_env(env=os.environ):
    env = dict(env)
    env.setdefault("AUTH0_CLIENT_ID", env.get("DEVICE_CLIENT_ID", ""))
    env.setdefault("AUTH0_CLIENT_SECRET", env.get("DEVICE_CLIENT_SECRET", ""))
    if env.get("AUTH0_CLIENT_ID"):
        missing = [k for k in ("AUTH0_DOMAIN", "AUTH0_CLIENT_SECRET", "AUTH0_AUDIENCE") if not env.get(k)]
        if missing:
            raise RuntimeError(f"missing environment variables: {missing}")
        return Auth0ClientCredentials(env["AUTH0_DOMAIN"], env["AUTH0_CLIENT_ID"],
                                      env["AUTH0_CLIENT_SECRET"], env["AUTH0_AUDIENCE"],
                                      cache_path=env.get("LASTSCENE_TOKEN_CACHE") or
                                      Path.home() / ".cache" / "lastscene" / "auth0.json")
    if env.get("LASTSCENE_DEVICE_TOKEN"):
        return StaticToken(env["LASTSCENE_DEVICE_TOKEN"])
    raise RuntimeError("no credentials: set AUTH0_CLIENT_ID/AUTH0_CLIENT_SECRET/AUTH0_DOMAIN/AUTH0_AUDIENCE "
                       "(or LASTSCENE_DEVICE_TOKEN for development)")
