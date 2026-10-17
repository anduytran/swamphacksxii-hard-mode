"""Train on a temporary Vultr server, publish the model, destroy the server. Built to never cost money.

    python -m lastscene_pipeline.vultr status            credit left, expiry, any trainer servers
    python -m lastscene_pipeline.vultr run [--gpu]       train once if the photos changed
    python -m lastscene_pipeline.vultr cleanup           destroy anything this tool created

Needs the pipeline's device credentials and VULTR_API_KEY, both in the --env-file (kept out of git); an
older ~/.lastscene/vultr.env (--vultr-env) still works. Turn on Vultr's API access control and allow only
your own IP for that key: it controls the whole account and is never sent to the server.

Guardrails, all checked before a server is created (any one failing means nothing is created):
  * not on or after the day before the credit expires (VULTR_CREDIT_EXPIRES, default 2026-11-17)
  * only when the photos changed since the published model (or --force)
  * the plan's worst case (hourly price x the time limit, rounded up to whole hours) is at most
    MAX_RUN_COST (default $2.00)
  * the account keeps at least MIN_CREDIT_LEFT (default $20.00) of credit after that worst case
  * no other trainer server exists (run cleanup first)
  * backups, IPv6, DDoS protection and activation emails off; no snapshots, block storage or reserved IPs
Then: the server powers itself off when training ends (or fails), and this tool destroys it as soon as
it stops, after --max-minutes (default: sized from the photo count, at least 60), on Ctrl-C, or on any error. Stopped Vultr servers still
bill; only destroying ends billing.
"""

import argparse
import base64
import math
import os
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from lastscene_pipeline import EXIT_CONFIG, EXIT_NETWORK, EXIT_NOTHING, EXIT_OK
from lastscene_pipeline.common import Api, PipelineError, load_env_file, run_main
from lastscene_pipeline.train import BACKGROUNDS

VULTR = "https://api.vultr.com/v2"
TAG = "lastscene-trainer"
# Vultr regions by distance from Gainesville, FL: atl Atlanta, mia Miami, dfw Dallas, ord Chicago, ewr New Jersey.
NEAR_FLORIDA = ("atl", "mia", "dfw", "ord", "ewr", "lax", "sjc", "sea")
REPO = "https://github.com/anduytran/swamphacksxii-hard-mode.git"
SETUP_MINUTES = 15  # boot, apt, pip (torch + ultralytics), sync, INT8 export, publish: an estimate
SECONDS_PER_IMAGE_EPOCH = 0.3  # 0.1 measured on an M2 with 4 threads (10 photos + 40 backgrounds, 100 epochs: 8.5 min); shared vCPUs ~3x slower
PIPELINE_VARS = ("API_URL", "AUTH0_DOMAIN", "AUTH0_AUDIENCE", "DEVICE_CLIENT_ID", "DEVICE_CLIENT_SECRET", "PAIRING_CODE")


def _money(env, key, default) -> float:
    return float(env.get(key) or default)


class Guardrails:
    def __init__(self, env=os.environ, today: date | None = None):
        self.expires = date.fromisoformat(env.get("VULTR_CREDIT_EXPIRES") or "2026-11-17")
        self.max_run_cost = _money(env, "MAX_RUN_COST", 2.00)
        self.min_credit_left = _money(env, "MIN_CREDIT_LEFT", 20.00)
        self.today = today or date.today()

    def check_date(self):
        if (self.expires - self.today).days <= 1:
            raise PipelineError(EXIT_CONFIG, f"refusing: the Vultr credit expires {self.expires}; after that any server would be charged to the card")

    def check_cost(self, hourly: float, max_minutes: int, credit_left: float) -> float:
        worst = round(hourly * math.ceil(max_minutes / 60), 2)
        if worst > self.max_run_cost:
            raise PipelineError(EXIT_CONFIG, f"refusing: this plan could cost ${worst:.2f} per run (limit MAX_RUN_COST=${self.max_run_cost:.2f})")
        if credit_left - worst < self.min_credit_left:
            raise PipelineError(EXIT_CONFIG, f"refusing: ${credit_left:.2f} credit left; a run could leave less than MIN_CREDIT_LEFT=${self.min_credit_left:.2f}")
        return worst


class Vultr:
    """The few Vultr API v2 calls this needs. The key is only ever sent to api.vultr.com."""

    def __init__(self, key: str, session=None, timeout: float = 30):
        if not key:
            raise PipelineError(EXIT_CONFIG, "VULTR_API_KEY is not set (add it to the --env-file)")
        self.http = session or requests.Session()
        self.headers = {"Authorization": f"Bearer {key}"}
        self.timeout = timeout

    def call(self, method: str, path: str, ok=(200,), **kwargs):
        try:
            r = self.http.request(method, VULTR + path, headers=self.headers, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise PipelineError(EXIT_NETWORK, f"Vultr {method} {path}: {type(exc).__name__}") from None
        if r.status_code not in ok:
            raise PipelineError(EXIT_NETWORK if r.status_code >= 500 or r.status_code == 429 else EXIT_CONFIG,
                                f"Vultr {method} {path}: {r.status_code} {r.text[:200]}")
        return r.json() if r.content else {}

    def credit_left(self) -> float:
        """Vultr shows credit as a negative balance; pending charges eat into it."""
        account = self.call("GET", "/account")["account"]
        return round(-float(account.get("balance") or 0) - float(account.get("pending_charges") or 0), 2)

    def trainers(self) -> list[dict]:
        return [i for i in self.call("GET", "/instances", params={"tag": TAG, "per_page": 100})["instances"] if TAG in (i.get("tags") or [])]

    def cheapest_plan(self, gpu: bool, region: str) -> tuple[dict, str]:
        """The cheapest fitting plan and the region to rent it in. For --gpu the region is whichever
        offers one, nearest Florida first (GPUs aren't in every region, or on every account)."""
        plans = self.call("GET", "/plans", params={"type": "vcg" if gpu else "all", "per_page": 500})["plans"]
        fit = [p for p in plans if p.get("type") == "vcg"] if gpu else [
            p for p in plans if p.get("vcpu_count", 0) >= 4 and p.get("ram", 0) >= 8192 and p.get("type") != "vcg"]
        if not gpu:
            here = [p for p in fit if region in (p.get("locations") or [])]
            if not here:
                raise PipelineError(EXIT_CONFIG, f"no 4 vCPU / 8 GB plan available in {region}")
            return min(here, key=hourly_price), region
        order = [region, *[r for r in NEAR_FLORIDA if r != region]]
        for r in order + sorted({loc for p in fit for loc in p.get("locations") or []} - set(order)):
            here = [p for p in fit if r in (p.get("locations") or [])]
            if here:
                return min(here, key=hourly_price), r
        cheapest = min(fit, key=hourly_price) if fit else None
        raise PipelineError(EXIT_CONFIG, "no region offers a GPU to this account" + (
            f" (the cheapest, {cheapest['id']}, is ${cheapest['monthly_cost']:,.2f}/month: above a $1,000/month account limit, "
            "Vultr hides it until support raises the limit)" if cheapest else "") + "; train on CPU (drop --gpu)")

    def ubuntu(self) -> int:
        oses = self.call("GET", "/os", params={"per_page": 500})["os"]
        for name in ("Ubuntu 24.04 LTS x64", "Ubuntu 22.04 LTS x64"):
            match = [o for o in oses if o.get("name") == name]
            if match:
                return match[0]["id"]
        raise PipelineError(EXIT_CONFIG, "no Ubuntu LTS image found")

    def create(self, region: str, plan: str, os_id: int, user_data: str) -> dict:
        body = {
            "region": region, "plan": plan, "os_id": os_id, "label": TAG, "hostname": TAG, "tags": [TAG],
            "user_data": base64.b64encode(user_data.encode()).decode(),
            "backups": "disabled", "enable_ipv6": False, "ddos_protection": False, "activation_email": False,
        }
        return self.call("POST", "/instances", ok=(202, 201, 200), json=body)["instance"]

    def instance(self, instance_id: str) -> dict | None:
        try:
            return self.call("GET", f"/instances/{instance_id}")["instance"]
        except PipelineError as exc:
            if " 404 " in str(exc):
                return None
            raise

    def destroy(self, instance_id: str):
        self.call("DELETE", f"/instances/{instance_id}", ok=(204, 200, 404))


def _state(instance: dict) -> str:
    return "/".join(str(instance.get(k)) for k in ("status", "power_status", "server_status"))


def _up(instance: dict | None) -> bool:
    """Booted and running. While Vultr sets a server up it flips between "running" and "stopped" with
    server_status "none", "locked" or "installingbooting"; only "ok" means it's ours (seen Oct 17)."""
    return bool(instance) and instance.get("power_status") == "running" and instance.get("server_status") == "ok"


def time_limit(photos: int, epochs: int, backgrounds: int = BACKGROUNDS) -> int:
    """Minutes before the server is destroyed: the estimated run plus half again, in whole billed hours (min 1).
    Training time grows with photos x epochs: 10 photos fit in 1 hour, 50 items x 30 photos need about 20."""
    estimate = SETUP_MINUTES + (photos + backgrounds) * epochs * SECONDS_PER_IMAGE_EPOCH / 60
    return max(1, math.ceil(estimate * 1.5 / 60)) * 60


def hourly_price(plan: dict) -> float:
    """Vultr bills by the hour up to the monthly price; 672 hours (28 days) is the conservative divisor."""
    return float(plan.get("hourly_cost") or plan["monthly_cost"] / 672)


def trainer_script(env, gpu: bool, ref: str, epochs: int, max_minutes: int) -> str:
    """cloud-init: install, sync, train, publish, then power off whatever happens (the runner destroys it)."""
    creds = "\n".join(f"{k}={env[k]}" for k in PIPELINE_VARS if env.get(k))
    torch = "pip install torch torchvision" if gpu else "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    device = "$(nvidia-smi >/dev/null 2>&1 && echo 0 || echo cpu)" if gpu else "cpu"  # no driver -> CPU, not a failed run
    return f"""#!/bin/bash
set -euo pipefail
exec > /var/log/lastscene-trainer.log 2>&1
trap 'shred -u /root/pipeline.env 2>/dev/null || true; poweroff' EXIT
shutdown -P +{max_minutes}
umask 077
cat > /root/pipeline.env <<'CREDS'
{creds}
CREDS
export DEBIAN_FRONTEND=noninteractive LASTSCENE_DATA=/opt/lastscene-data
apt-get update -q && apt-get install -yq python3-venv git libgl1 libglib2.0-0
git clone -q --depth 1 --branch {ref} {REPO} /opt/lastscene
cd /opt/lastscene/vision
python3 -m venv /opt/venv && . /opt/venv/bin/activate
pip install -q --upgrade pip && {torch} -q && pip install -q -r requirements.txt -r requirements-onnx.txt
python -m lastscene_pipeline.sync --env-file /root/pipeline.env --force
python -m lastscene_pipeline.train --imgsz 320 --epochs {epochs} --device {device} || [ $? -eq 3 ]
python -m lastscene_pipeline.publish --env-file /root/pipeline.env || [ $? -eq 3 ]
"""


class Runner:
    def __init__(self, vultr: Vultr, api: Api | None, guard: Guardrails, sleep=time.sleep, clock=time.monotonic, log=print):
        self.vultr, self.api, self.guard, self.sleep, self.clock, self.log = vultr, api, guard, sleep, clock, log

    def wanted(self, force: bool) -> str | None:
        """The dataset version to train, or None when the published model already matches the photos."""
        vocab = self.api.get_json("/api/v1/training/vocabulary")
        self.photos = sum(i.get("samples") or 0 for i in vocab["items"])
        if not self.photos:
            self.log("no Teach items photos yet: nothing to train")
            return None
        current = self.api.request("GET", "/api/v1/training/models/current", ok=(200, 404))
        if not force and current.status_code == 200 and current.json()["version"] == vocab["version"]:
            self.log(f"the published model already matches the photos ({vocab['version']})")
            return None
        return vocab["version"]

    def published(self, version: str) -> bool:
        r = self.api.request("GET", "/api/v1/training/models/current", ok=(200, 404))
        return r.status_code == 200 and r.json()["version"] == version

    def run(self, env, gpu=False, region="atl", ref="master", epochs=100, max_minutes=None, force=False, poll=30) -> int:
        self.guard.check_date()
        version = self.wanted(force)
        if version is None:
            return EXIT_NOTHING
        if self.vultr.trainers():
            raise PipelineError(EXIT_CONFIG, "refusing: a trainer server already exists; run `cleanup` first")
        max_minutes = max_minutes or time_limit(self.photos, epochs)
        plan, region = self.vultr.cheapest_plan(gpu, region)
        credit = self.vultr.credit_left()
        worst = self.guard.check_cost(hourly_price(plan), max_minutes, credit)
        self.log(f"plan {plan['id']} in {region} at ${hourly_price(plan):.3f}/h, {self.photos} photos, limit {max_minutes} min: "
                 f"worst case ${worst:.2f}, credit ${credit:.2f}")

        script = trainer_script(env, gpu, ref, epochs, max_minutes)
        instance = None
        started = self.clock()
        stop = {"now": False}
        previous = {s: signal.signal(s, lambda *_: stop.update(now=True)) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            instance = self.vultr.create(region, plan["id"], self.vultr.ubuntu(), script)
            self.log(f"created {instance['id']}; training dataset {version}")
            booted, seen = False, None
            while not stop["now"]:
                if self.published(version):
                    self.log("model published")
                    break
                state = self.vultr.instance(instance["id"])
                if state is not None and (now := _state(state)) != seen:
                    self.log(f"server {now}")
                    seen = now
                booted = booted or _up(state)
                if state is None or (booted and state.get("power_status") == "stopped"):
                    self.log("the server stopped")
                    break
                if self.clock() - started > max_minutes * 60:
                    self.log(f"time limit ({max_minutes} min) reached")
                    break
                self.sleep(poll)
        finally:
            if instance:
                self._destroy(instance["id"])
            for s, handler in previous.items():
                signal.signal(s, handler)
        minutes = (self.clock() - started) / 60
        ok = self.published(version)
        self.log(f"{'done' if ok else 'not published'} after {minutes:.0f} min, about ${hourly_price(plan) * math.ceil(max(minutes, 1) / 60):.2f}")
        return EXIT_OK if ok else EXIT_NETWORK

    def _destroy(self, instance_id: str):
        for attempt in range(5):
            try:
                self.vultr.destroy(instance_id)
                if self.vultr.instance(instance_id) is None:
                    self.log(f"destroyed {instance_id}")
                    return
            except PipelineError as exc:
                self.log(f"destroy attempt {attempt + 1} failed: {exc}")
            self.sleep(10)
        self.log(f"WARNING: could not confirm {instance_id} is destroyed; run `cleanup` and check the Vultr console")

    def cleanup(self) -> int:
        servers = self.vultr.trainers()
        for s in servers:
            self._destroy(s["id"])
        self.log(f"{len(servers)} trainer server(s) destroyed" if servers else "nothing to clean up")
        return EXIT_OK

    def status(self) -> int:
        credit = self.vultr.credit_left()
        servers = self.vultr.trainers()
        self.log(f"credit left ${credit:.2f}; expires {self.guard.expires}; trainer servers: {len(servers)}")
        for s in servers:
            self.log(f"  {s['id']} {s.get('status')} / {s.get('power_status')} since {s.get('date_created')}")
        return EXIT_OK


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "run", "cleanup"])
    p.add_argument("--env-file", help="device credentials and VULTR_API_KEY (KEY=VALUE, never commit)")
    p.add_argument("--vultr-env", default=str(Path.home() / ".lastscene" / "vultr.env"),
                   help="optional, older setup: a file holding only VULTR_API_KEY")
    p.add_argument("--gpu", action="store_true", help="a GPU plan instead of a 4 vCPU / 8 GB CPU one")
    p.add_argument("--region", default="atl", help="Vultr region id (atl = Atlanta); with --gpu, the nearest region offering one")
    p.add_argument("--ref", default="master", help="git branch the server trains from")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-minutes", type=int, help="default: estimated from the photo count (time_limit), at least 60")
    p.add_argument("--force", action="store_true", help="train even if the model matches the photos")
    args = p.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    if Path(args.vultr_env).is_file():
        load_env_file(args.vultr_env)
    runner = Runner(Vultr(os.environ.get("VULTR_API_KEY", "")), Api() if args.command == "run" else None, Guardrails())
    if args.command == "status":
        return runner.status()
    if args.command == "cleanup":
        return runner.cleanup()
    missing = [k for k in PIPELINE_VARS[1:] if not os.environ.get(k)]
    if missing:
        raise PipelineError(EXIT_CONFIG, f"missing {', '.join(missing)} (pass --env-file)")
    return runner.run(os.environ, args.gpu, args.region, args.ref, args.epochs, args.max_minutes, args.force)


if __name__ == "__main__":
    run_main(main)
