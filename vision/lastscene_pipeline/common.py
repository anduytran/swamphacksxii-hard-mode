"""Shared configuration: the same environment variables and device identity for sync, train and detect.

    API_URL               default https://www.lastscene.photo (the apex domain 308-redirects and drops the token)
    AUTH0_DOMAIN, AUTH0_AUDIENCE, DEVICE_CLIENT_ID, DEVICE_CLIENT_SECRET, PAIRING_CODE
    LASTSCENE_DATA        working folder for the dataset and models; default ./lastscene-data

The Auth0 token is cached to a file until it expires (lastscene_bridge.auth): the free plan caps M2M tokens.
"""

import json
import os
import sys
from pathlib import Path

import requests

from lastscene_pipeline import EXIT_CONFIG, EXIT_NETWORK

DEFAULT_API = "https://www.lastscene.photo"


class PipelineError(Exception):
    """Ends a script with an exit code from the spec and a one-line reason."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def data_dir(env=os.environ) -> Path:
    return Path(env.get("LASTSCENE_DATA") or "lastscene-data")


def load_env_file(path):
    """Optional KEY=VALUE file (keep it out of git). Values already in the environment win."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def api_base(env=os.environ) -> str:
    from lastscene_bridge.api_base import normalize_api_base

    return normalize_api_base(env.get("API_URL") or env.get("LASTSCENE_API_BASE") or DEFAULT_API)


def pairing_code(env=os.environ) -> str:
    return env.get("PAIRING_CODE") or env.get("LASTSCENE_PAIRING_CODE") or ""


class Api:
    """GET helper with the spec's rules: 401 -> one fresh token then stop, 403 -> missing device scope,
    404 on training routes -> device not in a household. No redirects are followed (they drop the token)."""

    def __init__(self, env=os.environ, tokens=None, session=None, timeout: float = 30):
        from lastscene_bridge.auth import token_provider_from_env

        self.base = api_base(env)
        self.pairing = pairing_code(env)
        try:
            self.tokens = tokens or token_provider_from_env(env)
        except RuntimeError as exc:
            raise PipelineError(EXIT_CONFIG, f"device credentials missing: {exc}") from None
        self.http = session or requests.Session()
        self.timeout = timeout

    def request(self, method: str, path: str, ok=(200,), **kwargs) -> requests.Response:
        """Any method, same token and error rules as get_json; returns the response when its status is in ok."""
        refreshed = False
        while True:
            try:
                headers = {"Authorization": f"Bearer {self.tokens.get()}", **kwargs.pop("headers", {})}
                if self.pairing:
                    headers["X-Pairing-Code"] = self.pairing
                r = self.http.request(method, self.base + path, headers=headers, timeout=self.timeout, allow_redirects=False, **kwargs)
            except requests.HTTPError as exc:
                raise PipelineError(EXIT_CONFIG, f"Auth0 token request failed: {exc}") from None
            except requests.RequestException as exc:
                raise PipelineError(EXIT_NETWORK, f"{method} {path}: {type(exc).__name__}") from None
            if r.status_code == 401 and not refreshed:
                self.tokens.invalidate()
                refreshed = True
                continue
            if r.status_code in ok:
                return r
            raise PipelineError(*_explain(path, r))

    def get_json(self, path: str) -> dict:
        refreshed = False
        while True:
            try:
                headers = {"Authorization": f"Bearer {self.tokens.get()}"}
                if self.pairing:
                    headers["X-Pairing-Code"] = self.pairing
                r = self.http.get(self.base + path, headers=headers, timeout=self.timeout, allow_redirects=False)
            except requests.HTTPError as exc:  # the Auth0 token request itself was refused
                raise PipelineError(EXIT_CONFIG, f"Auth0 token request failed: {exc}") from None
            except requests.RequestException as exc:
                raise PipelineError(EXIT_NETWORK, f"GET {path}: {type(exc).__name__}") from None
            if r.status_code == 401 and not refreshed:
                self.tokens.invalidate()
                refreshed = True
                continue
            if r.status_code == 200:
                return r.json()
            raise PipelineError(*_explain(path, r))


def _explain(path, r) -> tuple[int, str]:
    body = r.text[:200]
    if 300 <= r.status_code < 400:
        return EXIT_CONFIG, f"GET {path}: {r.status_code} redirect to {r.headers.get('Location')}; set API_URL to that host"
    if r.status_code == 401:
        return EXIT_CONFIG, f"GET {path}: 401 even with a fresh token (wrong client, secret or audience): {body}"
    if r.status_code == 403:
        return EXIT_CONFIG, f"GET {path}: 403, the client lacks the device scope (write:memories): {body}"
    if r.status_code == 404 and "/training/" in path:
        return EXIT_CONFIG, f"GET {path}: 404, this device isn't in a household yet (Household page -> add a wearable)"
    if r.status_code == 429 or r.status_code >= 500:
        return EXIT_NETWORK, f"GET {path}: {r.status_code} {body}"
    return EXIT_CONFIG, f"GET {path}: {r.status_code} {body}"


def set_current(models: Path, version: str):
    """Point models/current at models/<version> atomically (a one-line file; works without symlink rights)."""
    import shutil

    current = models / "current"
    if current.is_symlink() or current.is_dir():
        current.unlink() if current.is_symlink() else shutil.rmtree(current)
    tmp = models / ".current.tmp"
    tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(tmp, current)


def resolve_model_dir(path) -> Path | None:
    """models/current is a symlink or a one-line file naming the version folder (Windows can't always symlink)."""
    path = Path(path)
    if path.is_dir():
        return path
    if path.is_file():
        target = path.read_text(encoding="utf-8").strip()
        folder = (path.parent / target) if target else None
        return folder if folder and folder.is_dir() else None
    return None


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_main(main):
    """Call main(); map PipelineError to its exit code with the reason on stderr."""
    try:
        code = main()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        code = exc.code
    except KeyboardInterrupt:
        code = 0
    sys.exit(code or 0)
