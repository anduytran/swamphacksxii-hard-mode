"""publish -> the API -> fetch, against an in-memory stand-in for the model routes (no network)."""

import hashlib
import json
import re

import pytest

from lastscene_pipeline import EXIT_NETWORK, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline.common import Api, PipelineError
from lastscene_pipeline.fetch import run_fetch
from lastscene_pipeline.publish import run_publish

PART = 1000  # small parts, so the test files span several


class Resp:
    def __init__(self, status, body=b"", data=None):
        self.status_code, self.content, self._data = status, body, data
        self.text, self.headers = str(data or ""), {}

    def json(self):
        return self._data


class FakeModelApi:
    """The server's model routes: manifest, parts, complete (checks sha256), current, part download."""

    def __init__(self):
        self.models, self.parts, self.current, self.corrupt = {}, {}, None, False

    def request(self, method, url, headers=None, timeout=None, allow_redirects=False, data=None, json=None):
        path = url.split("testserver", 1)[1]
        if method == "GET" and path == "/api/v1/training/models/current":
            if not self.current:
                return Resp(404, data={"detail": "No trained model yet"})
            m = self.models[self.current]
            files = [{**f, "parts": -(-f["size"] // PART)} for f in m["files"]]
            return Resp(200, data={"id": self.current, "version": m["version"], "files": files, "part_bytes": PART})
        if method == "POST" and path == "/api/v1/training/models":
            mid = f"m{len(self.models)}"
            self.models[mid] = {**json, "complete": False}
            return Resp(201, data={"id": mid, "part_bytes": PART, "parts": {f["name"]: -(-f["size"] // PART) for f in json["files"]}})
        part = re.fullmatch(r"/api/v1/training/models/(\w+)/files/([\w.-]+)/parts/(\d+)", path)
        if part and method == "PUT":
            self.parts[(part[1], part[2], int(part[3]))] = data
            return Resp(204)
        if part and method == "GET":
            body = self.parts[(part[1], part[2], int(part[3]))]
            return Resp(200, body=b"X" + body[1:] if self.corrupt else body)
        done = re.fullmatch(r"/api/v1/training/models/(\w+)/complete", path)
        if done and method == "POST":
            m = self.models[done[1]]
            for f in m["files"]:
                got = b"".join(self.parts[k] for k in sorted(k for k in self.parts if k[:2] == (done[1], f["name"])))
                if hashlib.sha256(got).hexdigest() != f["sha256"]:
                    return Resp(422, data={"detail": "sha256 mismatch"})
            self.current = done[1]
            return Resp(200, data={"current": True})
        return Resp(404, data={"detail": f"no route {method} {path}"})


class Token:
    def get(self):
        return "t"

    def invalidate(self):
        pass


@pytest.fixture
def server():
    return FakeModelApi()


def client(server):
    return Api(env={"API_URL": "http://testserver"}, tokens=Token(), session=server)


def trained(tmp_path, version="v7", onnx=bytes(range(256)) * 9):
    folder = tmp_path / "models" / version
    folder.mkdir(parents=True)
    (folder / "model-int8.onnx").write_bytes(onnx)
    (folder / "classes.json").write_text(json.dumps({"classes": ["keys"]}))
    (folder / "metrics.json").write_text(json.dumps({"map50": 0.6, "imgsz": 320}))
    return folder


def test_publish_then_fetch_is_byte_for_byte(tmp_path, server):
    folder = trained(tmp_path)
    assert run_publish(folder, client(server), log=lambda m: None) == EXIT_OK
    assert run_publish(folder, client(server), log=lambda m: None) == EXIT_NOTHING  # same model: skipped

    pi = tmp_path / "pi"
    assert run_fetch(pi, client(server), log=lambda m: None) == EXIT_OK
    assert (pi / "current").read_text().strip() == "v7"
    assert (pi / "v7" / "model-int8.onnx").read_bytes() == (folder / "model-int8.onnx").read_bytes()
    assert run_fetch(pi, client(server), log=lambda m: None) == EXIT_NOTHING  # already current


def test_fetch_with_nothing_published_and_corrupt_download(tmp_path, server):
    assert run_fetch(tmp_path / "pi", client(server), log=lambda m: None) == EXIT_NOTHING
    run_publish(trained(tmp_path), client(server), log=lambda m: None)
    server.corrupt = True
    with pytest.raises(PipelineError) as e:
        run_fetch(tmp_path / "pi", client(server), log=lambda m: None)
    assert e.value.code == EXIT_NETWORK
    assert not (tmp_path / "pi" / "current").exists()  # a bad download never becomes current


def test_publish_needs_a_trained_model(tmp_path, server):
    with pytest.raises(PipelineError):
        run_publish(tmp_path / "models" / "current", client(server), log=lambda m: None)
