"""The Vultr runner's guardrails, against a fake Vultr API: nothing here touches a real account."""

import base64
import signal
import socket
import threading
from datetime import date

import pytest
import requests
import urllib3.util.connection
from urllib3.connection import HTTPSConnection

from lastscene_pipeline import EXIT_CONFIG, EXIT_NETWORK, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline import vultr as vultr_module
from lastscene_pipeline.common import PipelineError
from lastscene_pipeline.vultr import TAG, Guardrails, Runner, Vultr, trainer_script

ENV = {"API_URL": "https://www.lastscene.photo", "AUTH0_DOMAIN": "d", "AUTH0_AUDIENCE": "a",
       "DEVICE_CLIENT_ID": "id", "DEVICE_CLIENT_SECRET": "s3cret", "PAIRING_CODE": "AB12"}


class R:
    def __init__(self, status, data=None):
        self.status_code, self._data = status, data
        self.content = b"x" if data is not None else b""
        self.text = str(data)

    def json(self):
        return self._data


class FakeVultr:
    """Vultr API v2 as far as the runner uses it. `stops_after` polls, the server powers itself off."""

    def __init__(self, credit=100.0, pending=0.0, hourly=0.06, existing=(), stops_after=2, fail_create=False, gpu_regions=(),
                 boots_after=1):
        self.gpu_regions = list(gpu_regions)
        self.credit, self.pending, self.hourly = credit, pending, hourly
        self.instances = {i["id"]: i for i in existing}
        self.created, self.deleted, self.polls = [], [], 0
        self.stops_after, self.fail_create, self.boots_after = stops_after, fail_create, boots_after

    def request(self, method, url, headers=None, timeout=None, params=None, json=None):
        assert url.startswith("https://api.vultr.com/v2/") and headers["Authorization"] == "Bearer KEY"
        path = url.removeprefix("https://api.vultr.com/v2")
        if path == "/account":
            return R(200, {"account": {"balance": -self.credit, "pending_charges": self.pending}})
        if path == "/instances" and method == "GET":
            return R(200, {"instances": [i for i in self.instances.values() if TAG in i["tags"]]})
        if path == "/plans":
            return R(200, {"plans": [
                {"id": "vc2-1c-1gb", "type": "vc2", "vcpu_count": 1, "ram": 1024, "monthly_cost": 5, "locations": ["atl"]},
                {"id": "vhp-4c-8gb", "type": "vhp", "vcpu_count": 4, "ram": 8192, "hourly_cost": self.hourly, "monthly_cost": 48, "locations": ["atl"]},
                {"id": "vhp-8c-16gb", "type": "vhp", "vcpu_count": 8, "ram": 16384, "hourly_cost": self.hourly * 2, "monthly_cost": 96, "locations": ["atl"]},
                {"id": "vcg-l40s-16c", "type": "vcg", "vcpu_count": 16, "ram": 184320, "hourly_cost": 1.671, "monthly_cost": 1122.91,
                 "locations": self.gpu_regions},
            ]})
        if path == "/os":
            return R(200, {"os": [{"id": 2284, "name": "Ubuntu 24.04 LTS x64"}]})
        if path == "/instances" and method == "POST":
            if self.fail_create:
                return R(500, {"error": "boom"})
            # as the real API: "running" when created, then "stopped" and locked while Vultr installs it
            inst = {"id": f"i{len(self.created)}", "tags": json["tags"], "status": "pending", "power_status": "running",
                    "server_status": "none"}
            self.created.append(json)
            self.instances[inst["id"]] = inst
            return R(202, {"instance": inst})
        if path.startswith("/instances/"):
            iid = path.split("/")[2]
            if method == "DELETE":
                self.deleted.append(iid)
                self.instances.pop(iid, None)
                return R(204)
            if iid not in self.instances:
                return R(404, {"error": "not found"})
            self.polls += 1
            inst = self.instances[iid]
            if self.polls >= self.stops_after:
                inst.update(status="active", power_status="stopped", server_status="ok")
            elif self.polls >= self.boots_after:
                inst.update(status="active", power_status="running", server_status="ok")
            elif self.polls > 1:  # the first poll still shows the "running" it was created with
                inst.update(status="active", power_status="stopped", server_status="locked")
            return R(200, {"instance": inst})
        raise AssertionError(f"unexpected {method} {path}")


class Pipeline:
    """The LastScene API side: vocabulary and the current model."""

    def __init__(self, version="v2", photos=10, current=None, publishes=False):
        self.version, self.photos, self.current, self.publishes = version, photos, current, publishes
        self.calls = 0

    def get_json(self, path):
        return {"version": self.version, "items": [{"name": "keys", "samples": self.photos}]}

    def request(self, method, path, ok=(200,)):
        self.calls += 1
        if self.publishes and self.calls > 2:  # the server published while we polled
            self.current = self.version
        return R(200, {"version": self.current}) if self.current else R(404, {"detail": "none"})


def runner(fake, pipeline, today=date(2026, 10, 18), clock=None):
    ticks = iter(range(0, 10**6, 30))
    return Runner(Vultr("KEY", session=fake), pipeline, Guardrails({}, today=today),
                  sleep=lambda s: None, clock=clock or (lambda: next(ticks)), log=lambda m: None)


def test_trains_publishes_and_always_destroys():
    fake = FakeVultr()
    code = runner(fake, Pipeline(publishes=True)).run(ENV)
    assert code == EXIT_OK
    (body,) = fake.created
    assert body["plan"] == "vhp-4c-8gb"  # the cheapest plan that fits, not the bigger one
    assert body["backups"] == "disabled" and body["enable_ipv6"] is False and body["ddos_protection"] is False
    assert body["tags"] == [TAG] and fake.deleted == ["i0"] and not fake.instances


def test_destroys_when_the_server_stops_without_publishing():
    fake = FakeVultr(stops_after=2)
    assert runner(fake, Pipeline()).run(ENV) == EXIT_NETWORK
    assert fake.deleted == ["i0"] and not fake.instances


def test_destroys_at_the_time_limit():
    fake = FakeVultr(stops_after=10**6)  # never stops by itself
    assert runner(fake, Pipeline()).run(ENV, max_minutes=5) == EXIT_NETWORK
    assert fake.deleted == ["i0"]


def test_destroys_when_something_crashes_mid_run():
    fake = FakeVultr(stops_after=10**6)

    class Boom(Pipeline):
        def request(self, method, path, ok=(200,)):
            self.calls += 1
            if self.calls > 2:
                raise RuntimeError("network fell over")
            return R(404, {})

    with pytest.raises(RuntimeError):
        runner(fake, Boom()).run(ENV)
    assert fake.deleted == ["i0"] and not fake.instances


def test_destroys_on_ctrl_c():
    fake = FakeVultr(stops_after=10**6)
    r = runner(fake, Pipeline())
    r.sleep = lambda s: signal.raise_signal(signal.SIGINT)  # Ctrl-C while waiting
    r.run(ENV)
    assert fake.deleted == ["i0"]


@pytest.mark.parametrize(
    "fake, pipeline, today, reason",
    [
        (FakeVultr(), Pipeline(), date(2026, 11, 16), "expires"),  # the day before the credit expires
        (FakeVultr(), Pipeline(), date(2026, 12, 1), "expires"),
        (FakeVultr(credit=20.03), Pipeline(), date(2026, 10, 18), "MIN_CREDIT_LEFT"),
        (FakeVultr(credit=100, pending=80), Pipeline(), date(2026, 10, 18), "MIN_CREDIT_LEFT"),  # pending charges count
        (FakeVultr(hourly=2.5), Pipeline(), date(2026, 10, 18), "MAX_RUN_COST"),
        (FakeVultr(existing=[{"id": "old", "tags": [TAG]}]), Pipeline(), date(2026, 10, 18), "already exists"),
    ],
)
def test_guardrails_refuse_before_creating_anything(fake, pipeline, today, reason):
    with pytest.raises(PipelineError) as e:
        runner(fake, pipeline, today=today).run(ENV)
    assert reason in str(e.value)
    assert fake.created == []


def test_nothing_to_do_creates_nothing():
    fake = FakeVultr()
    assert runner(fake, Pipeline(current="v2")).run(ENV) == EXIT_NOTHING  # model matches the photos
    assert runner(fake, Pipeline(photos=0)).run(ENV) == EXIT_NOTHING  # no photos yet
    assert fake.created == []


def test_create_failure_leaves_nothing_behind():
    fake = FakeVultr(fail_create=True)
    with pytest.raises(PipelineError):
        runner(fake, Pipeline()).run(ENV)
    assert fake.instances == {}


def test_cleanup_destroys_only_trainer_servers():
    fake = FakeVultr(existing=[{"id": "t1", "tags": [TAG]}, {"id": "web", "tags": ["website"]}])
    runner(fake, Pipeline()).cleanup()
    assert fake.deleted == ["t1"] and "web" in fake.instances


def test_server_script_powers_off_whatever_happens_and_never_holds_the_vultr_key():
    script = trainer_script({**ENV, "VULTR_API_KEY": "KEY"}, gpu=False, ref="master", epochs=50, max_minutes=45)
    assert "trap 'shred -u /root/pipeline.env 2>/dev/null || true; poweroff' EXIT" in script
    assert "shutdown -P +45" in script
    assert "VULTR_API_KEY" not in script and "KEY\n" not in script
    assert "DEVICE_CLIENT_SECRET=s3cret" in script  # the device secret is shredded at the end
    assert "--device cpu" in script and "whl/cpu" in script
    base64.b64encode(script.encode())  # what Vultr receives


def test_gpu_picks_the_nearest_region_that_has_one():
    fake = FakeVultr(gpu_regions=["ewr", "mia"])
    plan, region = Vultr("KEY", session=fake).cheapest_plan(True, "atl")
    assert plan["id"] == "vcg-l40s-16c" and region == "mia"  # Miami before New Jersey
    assert runner(fake, Pipeline(publishes=True)).run(ENV, gpu=True) == EXIT_OK
    assert fake.created[0]["region"] == "mia" and fake.deleted == ["i0"]


def test_gpu_unavailable_explains_and_creates_nothing():
    fake = FakeVultr(gpu_regions=[])
    with pytest.raises(PipelineError) as e:
        runner(fake, Pipeline()).run(ENV, gpu=True)
    assert "no region offers a GPU" in str(e.value) and "$1,122.91/month" in str(e.value)
    assert fake.created == []


def test_gpu_script_falls_back_to_cpu_without_a_driver():
    script = trainer_script(ENV, gpu=True, ref="master", epochs=50, max_minutes=45)
    assert "nvidia-smi >/dev/null 2>&1 && echo 0 || echo cpu" in script


def test_a_server_still_installing_is_not_mistaken_for_one_that_stopped():
    # Vultr reports a new server as active + stopped while installing it: an early version of the runner
    # destroyed it right there (seen on a real benchmark run, Oct 17).
    fake = FakeVultr(boots_after=3, stops_after=6)
    r, logs = runner(fake, Pipeline()), []
    r.log = logs.append
    assert r.run(ENV) == EXIT_NETWORK
    assert fake.polls >= 6 and "the server stopped" in logs and fake.deleted == ["i0"]


def test_time_limit_grows_with_photos_in_whole_billed_hours():
    from lastscene_pipeline.vultr import time_limit

    assert time_limit(10, 100) == 60  # the 10-photo Pin set: about 40 min estimated
    assert time_limit(100, 100) == 180  # about 85 min estimated, plus half again
    assert time_limit(1500, 100) == 1200  # 50 items x 30 photos: 20 h, about $1.10 at $0.055/h
    assert all(time_limit(n, 100) % 60 == 0 for n in range(0, 2000, 37))


def test_the_time_limit_follows_the_photo_count_into_the_server_script():
    fake = FakeVultr(stops_after=1)
    runner(fake, Pipeline(photos=100)).run(ENV)
    import base64

    assert "shutdown -P +180" in base64.b64decode(fake.created[0]["user_data"]).decode()


def test_vultr_calls_go_over_ipv4_even_when_the_machine_has_ipv6(monkeypatch):
    # On Oct 17 the Pi on a Verizon hotspot reached Vultr over IPv6 and got 401 "Unauthorized IP address:
    # 2600:1006:...", although the hotspot's IPv4 address was allowed. api.vultr.test has both, like api.vultr.com.
    lookups, real = [], socket.getaddrinfo

    def dns(host, port, family=0, type=0, proto=0, flags=0):
        if host != "api.vultr.test":
            return real(host, port, family, type, proto, flags)
        lookups.append(family)
        v4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        v6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port, 0, 0))
        return [v4] if family == socket.AF_INET else [v6, v4]  # IPv6 first, as resolvers usually answer

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    monkeypatch.setattr(urllib3.util.connection, "HAS_IPV6", True)
    peers = []
    with socket.create_server(("127.0.0.1", 0)) as server:  # hangs up before TLS: no request is ever sent
        server.settimeout(5)
        port = server.getsockname()[1]
        monkeypatch.setattr(vultr_module, "VULTR", f"https://api.vultr.test:{port}/v2")

        def hang_up():
            conn, (ip, _) = server.accept()
            peers.append(ip)
            conn.close()

        listener = threading.Thread(target=hang_up, daemon=True)
        listener.start()
        vultr = Vultr("KEY", timeout=5)
        vultr.http.trust_env = False  # no proxy from the environment
        with pytest.raises(PipelineError) as e:
            vultr.credit_left()
        listener.join(5)
    assert e.value.code == EXIT_NETWORK
    assert lookups == [socket.AF_INET] and peers == ["127.0.0.1"]
    assert urllib3.util.connection.allowed_gai_family() == socket.AF_UNSPEC  # the rest of the process keeps IPv6


def test_only_the_vultr_session_is_ipv4_only():
    def connection_class(session):
        pool = session.get_adapter("https://api.vultr.com").poolmanager.connection_from_host("api.vultr.com", 443, scheme="https")
        return pool.ConnectionCls

    assert connection_class(Vultr("KEY").http) is not HTTPSConnection
    assert issubclass(connection_class(Vultr("KEY").http), HTTPSConnection)
    assert connection_class(Vultr("KEY", ipv4_only=False).http) is HTTPSConnection  # VULTR_IPV4_ONLY=0
    assert connection_class(requests.Session()) is HTTPSConnection  # e.g. the LastScene API calls


class Refusing:
    """Vultr answering every call with one error, as with API access control on and the address not allowed."""

    def __init__(self, status, body):
        self.status, self.body = status, body

    def request(self, method, url, **kwargs):
        return R(self.status, self.body)


@pytest.mark.parametrize("address", ["2001:db8:4f1a::1c", "203.0.113.7"])
def test_a_refused_address_is_named_with_where_to_allow_it(address):
    vultr = Vultr("KEY", session=Refusing(401, {"error": f"Unauthorized IP address: {address}", "status": 401}))
    with pytest.raises(PipelineError) as e:
        vultr.credit_left()
    assert e.value.code == EXIT_CONFIG
    assert f"401, the API key isn't allowed from {address} (the address Vultr saw)" in str(e.value)
    assert "Account -> API -> Access Control" in str(e.value)


def test_a_refused_address_without_one_in_the_reply_still_says_where_to_allow_it():
    vultr = Vultr("KEY", session=Refusing(401, {"error": "Unauthorized IP address", "status": 401}))
    with pytest.raises(PipelineError) as e:
        vultr.trainers()
    assert "add this machine's address under Account -> API -> Access Control" in str(e.value)


def test_a_wrong_key_is_not_reported_as_a_refused_address():
    vultr = Vultr("KEY", session=Refusing(401, {"error": "Invalid API token.", "status": 401}))
    with pytest.raises(PipelineError) as e:
        vultr.credit_left()
    assert e.value.code == EXIT_CONFIG and "401" in str(e.value) and "Access Control" not in str(e.value)


def test_skipped_items_stay_out_of_the_photo_count_and_reach_the_server_script():
    import base64

    class TwoItems(Pipeline):
        def get_json(self, path):
            return {"version": self.version, "items": [{"name": "keys", "samples": 10}, {"name": "Pin", "samples": 1000}]}

    fake = FakeVultr(stops_after=1)
    run = runner(fake, TwoItems())
    run.run(ENV, skip=["pin"])
    assert run.photos == 10
    script = base64.b64decode(fake.created[0]["user_data"]).decode()
    assert "sync --env-file /root/pipeline.env --force --skip pin\n" in script and "shutdown -P +60" in script
