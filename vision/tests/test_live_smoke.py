"""scripts/live_smoke.py helpers: JWT claims, base URL, last-seen comparison, reopen-test timing. No network."""

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from scripts import live_smoke as smoke

OBS = {
    "observation_id": "c8e00646-d873-5c5d-95f4-c088fa0be89a",
    "captured_at": "2026-10-17T05:36:00.003Z",
    "source": "auto",
    "image_size": {"width": 1920, "height": 1080},
    "objects": [{"label": "keys", "confidence": 0.054, "bbox": [0.3797, 0.637, 0.0885, 0.1565],
                 "track_id": "mem_bunch-of-keys_30fa4cad", "event": "seen"}],
}


def jwt_with(claims: dict) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{enc({'alg': 'RS256', 'typ': 'JWT'})}.{enc(claims)}.c2lnbmF0dXJl"


def device_claims(**kw):
    return {"iss": "https://tenant.auth0.com/", "aud": "https://api.lastscene", "sub": "abc@clients",
            "scope": "write:memories", "gty": "client-credentials", "exp": 2_000_000_000, **kw}


def stored(**kw):
    """What GET /items/keys/last-seen returns for OBS (float32 storage: tiny rounding noise)."""
    return {"label": "keys", "observation_id": OBS["observation_id"], "captured_at": "2026-10-17T05:36:00.003000+00:00",
            "bbox": [0.37970001, 0.63699999, 0.0885, 0.1565], "confidence": 0.054, "maybe": True,
            "image_size": {"width": 1920, "height": 1080}, "found_at": None, **kw}


# ------------------------------------------------------------------ JWT claims

def test_decodes_payload_without_verifying_and_handles_missing_padding():
    claims = device_claims(scope="read:memories write:memories", x="é" * 5)  # odd lengths need "=" padding
    assert smoke.decode_jwt_claims(jwt_with(claims)) == claims


@pytest.mark.parametrize("token", ["", "abc", "a.b", "a.b.c.d"])
def test_rejects_non_jwt(token):
    with pytest.raises(ValueError):
        smoke.decode_jwt_claims(token)


def test_device_token_has_no_problems():
    assert smoke.claim_problems(device_claims(), "https://api.lastscene", now=1_900_000_000) == []
    # aud may be a list; a sub ending in @clients also counts as machine-to-machine
    assert smoke.claim_problems(device_claims(aud=["https://api.lastscene", "x"], gty=None),
                                "https://api.lastscene", now=1_900_000_000) == []


def test_missing_scope_wrong_audience_user_token_and_expiry_are_reported():
    problems = smoke.claim_problems(device_claims(scope="read:memories", aud="https://other", gty="password",
                                                  sub="auth0|user", exp=100), "https://api.lastscene", now=200)
    text = " | ".join(problems)
    assert "no write:memories" in text and "aud" in text and "machine-to-machine" in text and "expired" in text
    assert smoke.claim_problems(device_claims(scope=""), None, now=0) == ["scope has no write:memories (got none)"]


# ------------------------------------------------------------------ base URL

@pytest.mark.parametrize("raw, base", [
    (None, "https://www.lastscene.photo"),
    ("", "https://www.lastscene.photo"),
    ("https://lastscene.photo", "https://www.lastscene.photo"),        # apex 308-redirects (drops the token)
    ("https://lastscene.photo/", "https://www.lastscene.photo"),
    ("lastscene.photo", "https://www.lastscene.photo"),
    ("HTTPS://LastScene.Photo/api/v1/", "https://www.lastscene.photo"),
    ("https://www.lastscene.photo", "https://www.lastscene.photo"),
    ("http://127.0.0.1:8765/", "http://127.0.0.1:8765"),
    ("https://lastscene.photo:8443", "https://www.lastscene.photo:8443"),
    ("https://notlastscene.photo", "https://notlastscene.photo"),
])
def test_normalize_base(raw, base):
    assert smoke.normalize_base(raw) == base


# ------------------------------------------------------------------ last-seen comparison

def test_same_instant_across_formats_and_naive_times_are_refused():
    assert smoke.same_instant("2026-10-17T05:36:00.003Z", "2026-10-17T05:36:00.003000+00:00")
    assert smoke.same_instant("2026-10-17T05:36:00.003Z", "2026-10-17T01:36:00.003-04:00")
    assert not smoke.same_instant("2026-10-17T05:36:00.003Z", "2026-10-17T05:36:00.004Z")
    with pytest.raises(ValueError):
        smoke.parse_time("2026-10-17T05:36:00")


def test_bbox_tolerance():
    assert smoke.bbox_matches([0.1, 0.2, 0.3, 0.4], [0.10009, 0.2, 0.3, 0.39991])
    assert not smoke.bbox_matches([0.1, 0.2, 0.3, 0.4], [0.1002, 0.2, 0.3, 0.4])
    assert not smoke.bbox_matches([0.1, 0.2, 0.3, 0.4], None)
    assert not smoke.bbox_matches([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3])


def test_matching_last_seen_has_no_problems():
    assert smoke.last_seen_problems(OBS, stored()) == []


@pytest.mark.parametrize("change, word", [
    ({"bbox": [0.5, 0.637, 0.0885, 0.1565]}, "bbox"),
    ({"captured_at": "2026-10-17T05:36:01.003Z"}, "captured_at"),
    ({"confidence": 0.35}, "confidence"),          # inflated confidence is a failure
    ({"image_size": {"width": 960, "height": 540}}, "image_size"),
    ({"maybe": None}, "maybe"),
    ({"observation_id": str(uuid.uuid4())}, "observation_id"),
])
def test_last_seen_differences_are_reported(change, word):
    problems = smoke.last_seen_problems(OBS, stored(**change))
    assert len(problems) == 1 and problems[0].startswith(word)


# ------------------------------------------------------------------ reopen test

def test_reopen_capture_time_is_strictly_between_on_a_whole_millisecond():
    lo = datetime(2026, 10, 17, 5, 36, 0, 3000, tzinfo=timezone.utc)
    hi = datetime(2026, 10, 17, 11, 0, 0, 1000, tzinfo=timezone.utc)
    t = smoke.pick_reopen_captured_at(lo, hi)
    assert lo < t < hi and t.microsecond % 1000 == 0 and t.tzinfo is not None
    # 2 ms apart: the only choice is the middle millisecond; 1 ms apart: nothing fits
    assert smoke.pick_reopen_captured_at(lo, lo + timedelta(milliseconds=2)) == lo + timedelta(milliseconds=1)
    with pytest.raises(ValueError):
        smoke.pick_reopen_captured_at(lo, lo + timedelta(milliseconds=1))
    with pytest.raises(ValueError):
        smoke.pick_reopen_captured_at(hi, lo)


def test_reopen_capture_time_accepts_other_offsets():
    lo = datetime(2026, 10, 17, 1, 36, tzinfo=timezone(timedelta(hours=-4)))  # 05:36Z
    hi = datetime(2026, 10, 17, 5, 38, tzinfo=timezone.utc)
    assert smoke.iso_ms(smoke.pick_reopen_captured_at(lo, hi)) == "2026-10-17T05:37:00.000Z"


def test_reopen_observation_id_is_stable_new_and_a_uuid5():
    rid = smoke.reopen_observation_id(OBS["observation_id"])
    assert rid == smoke.reopen_observation_id(OBS["observation_id"]) != OBS["observation_id"]
    assert uuid.UUID(rid).version == 5


# ------------------------------------------------------------------ plumbing without a network

class FakeResponse:
    def __init__(self, status, headers=None, body=None):
        self.status_code, self.headers, self._body = status, headers or {}, body or {}
        self.text, self.reason = json.dumps(self._body), ""

    def json(self):
        return self._body


class FakeHTTP:
    def __init__(self, response):
        self.response, self.calls = response, []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self.response


def test_redirects_are_not_followed_and_fail_with_the_fix(capsys):
    s = smoke.Smoke("https://www.lastscene.photo", {}, OBS, b"jpeg")
    s.token = "secret-token"
    s.http = FakeHTTP(FakeResponse(308, {"Location": "https://elsewhere/api/v1/items/last-seen"}))
    with pytest.raises(smoke.StepFailed):
        s.send("1 auth", "GET", s.url("/api/v1/items/last-seen"))
    method, url, kw = s.http.calls[0]
    assert kw["allow_redirects"] is False and kw["headers"]["Authorization"] == "Bearer secret-token"
    out = capsys.readouterr().out
    assert "FAIL" in out and "308" in out and "secret-token" not in out


def test_low_confidence_422_names_the_deployment_fix(capsys):
    s = smoke.Smoke("https://www.lastscene.photo", {"PAIRING_CODE": "AB12-CD34"}, OBS, b"jpeg")
    s.http = FakeHTTP(FakeResponse(422, body={"detail": "no object at or above confidence 0.35"}))
    with pytest.raises(smoke.StepFailed):
        s.step_post()
    _, _, kw = s.http.calls[0]
    assert kw["headers"]["X-Pairing-Code"] == "AB12-CD34"
    assert json.loads(kw["data"]["observation"]) == OBS and kw["files"]["image"][2] == "image/jpeg"
    assert "MIN_DETECTION_CONFIDENCE=0.05" in capsys.readouterr().out


def test_schema_422_on_confidence_is_not_blamed_on_the_floor(capsys):
    s = smoke.Smoke("https://www.lastscene.photo", {"PAIRING_CODE": "AB12-CD34"}, OBS, b"jpeg")
    s.http = FakeHTTP(FakeResponse(422, body={"detail": "observation: objects.0.confidence: Input should be "
                                                        "less than or equal to 1"}))
    with pytest.raises(smoke.StepFailed):
        s.step_post()
    out = capsys.readouterr().out
    assert "MIN_DETECTION_CONFIDENCE" not in out and "breaks the contract" in out


def test_signed_link_query_is_not_printed_on_errors(capsys):
    s = smoke.Smoke("https://www.lastscene.photo", {}, OBS, b"jpeg")
    s.http = FakeHTTP(FakeResponse(302, {"Location": "https://elsewhere/"}))
    with pytest.raises(smoke.StepFailed):
        s.send("3 photo", "GET", "https://www.lastscene.photo/api/v1/images/ls-1?exp=1&sig=SIGNATURE", auth=False)
    out = capsys.readouterr().out
    assert "/api/v1/images/ls-1" in out and "SIGNATURE" not in out


class SeqHTTP(FakeHTTP):
    def __init__(self, *responses):
        super().__init__(None)
        self.responses = list(responses)

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self.responses.pop(0)


def test_leftover_reopen_record_fails_step_3_unless_reopen_test_may_delete_it(capsys):
    rid = smoke.reopen_observation_id(OBS["observation_id"])
    leftover = stored(observation_id=rid, captured_at="2026-10-17T08:00:00Z")
    s = smoke.Smoke("https://www.lastscene.photo", {}, OBS, b"jpeg")
    s.http = SeqHTTP(FakeResponse(200, body=leftover))
    with pytest.raises(smoke.StepFailed):
        s.step_last_seen()
    assert [c[0] for c in s.http.calls] == ["GET"] and "--reopen-test" in capsys.readouterr().out

    import cv2
    import numpy as np
    jpeg = cv2.imencode(".jpg", np.zeros((1080, 1920, 3), np.uint8))[1].tobytes()
    photo = FakeResponse(200, {"Content-Type": "image/jpeg"})
    photo.content = jpeg
    s = smoke.Smoke("https://www.lastscene.photo", {}, OBS, jpeg, reopen_test=True)
    s.http = SeqHTTP(FakeResponse(200, body=leftover), FakeResponse(200, body={"deleted": rid}),
                     FakeResponse(200, body=stored(image_url="https://www.lastscene.photo/api/v1/images/ls-1?sig=x")),
                     photo)
    s.step_last_seen()
    methods = [(c[0], c[1].split("/api/v1/")[1]) for c in s.http.calls]
    assert methods[:3] == [("GET", "items/keys/last-seen"), ("DELETE", f"observations/{rid}"),
                           ("GET", "items/keys/last-seen")]
    assert s.last_seen["observation_id"] == OBS["observation_id"] and s.failed == 0


def test_dashboard_hint_warns_when_older_than_the_timeline_window(capsys):
    s = smoke.Smoke("https://www.lastscene.photo", {"DEVICE_CLIENT_ID": "cid"}, OBS, b"jpeg")
    s.last_seen = stored(captured_at=smoke.iso_ms(datetime.now(timezone.utc) - timedelta(hours=30)),
                         bbox=[0.3797, 0.637, 0.0885, 0.1565])
    s.step_dashboard_hint()
    out = capsys.readouterr().out
    assert "cid" in out and "[0.3797, 0.637, 0.0885, 0.1565]" in out and "only shows the last 24 h" in out
    s.last_seen = stored(captured_at=smoke.iso_ms(datetime.now(timezone.utc) - timedelta(hours=1)))
    s.step_dashboard_hint()
    assert "WARN" not in capsys.readouterr().out


def test_reopen_needs_found_and_found_needs_an_observation():
    with pytest.raises(SystemExit):
        smoke.main(["x.json", "--reopen-test"])
    with pytest.raises(SystemExit):
        smoke.main(["--found"])


def test_mask_keeps_only_the_last_two_characters():
    assert smoke.mask("AB12-CD34-EF56").endswith("56") and "AB12" not in smoke.mask("AB12-CD34-EF56")
    assert smoke.mask("") == "(not set)"
