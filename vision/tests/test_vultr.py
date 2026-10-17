"""The Vultr runner's guardrails, against a fake Vultr API: nothing here touches a real account."""

import base64
import signal
from datetime import date

import pytest

from lastscene_pipeline import EXIT_NETWORK, EXIT_NOTHING, EXIT_OK
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

    def __init__(self, credit=100.0, pending=0.0, hourly=0.06, existing=(), stops_after=2, fail_create=False):
        self.credit, self.pending, self.hourly = credit, pending, hourly
        self.instances = {i["id"]: i for i in existing}
        self.created, self.deleted, self.polls = [], [], 0
        self.stops_after, self.fail_create = stops_after, fail_create

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
            ]})
        if path == "/os":
            return R(200, {"os": [{"id": 2284, "name": "Ubuntu 24.04 LTS x64"}]})
        if path == "/instances" and method == "POST":
            if self.fail_create:
                return R(500, {"error": "boom"})
            inst = {"id": f"i{len(self.created)}", "tags": json["tags"], "status": "pending", "power_status": "running"}
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
                inst.update(status="active", power_status="stopped")
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
