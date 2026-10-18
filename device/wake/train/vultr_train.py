"""Rent one Vultr 32-core CPU server (paid from credit, ~$0.88/h) for the wake-model retrain, run remote_train.sh on it,
fetch the model, then destroy the server. It runs until training finishes (about 40 min on Oct 18), stopping early only
if the credit falls below CREDIT_FLOOR; a failed run keeps the server (and prints its address) so it can be fixed and
resumed, since each stage skips what's done. The API key is read from ~/.lastscene/vultr.env and never printed.

    python device/wake/train/vultr_train.py     # needs the training venv's packages for the Vultr client (requests)"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vision"))
from lastscene_pipeline.common import load_env_file  # noqa: E402
from lastscene_pipeline.vultr import Guardrails, Vultr, _up  # noqa: E402

PLAN, PRICE, HARD_STOP_MIN = "voc-c-32c-64gb-500s-amd", 0.877, 55  # the time limit covers booting only
CREDIT_FLOOR = 5.00
REGIONS = ("atl", "mia", "ewr", "ord", "dfw", "lax")
HERE = Path(__file__).parent
WORK = Path.home() / ".lastscene" / "wake_vultr"  # the throwaway SSH key and the results; outside the repo
WORK.mkdir(parents=True, exist_ok=True)
KEY = WORK / "id_ed25519"
KNOWN = WORK / "known_hosts"
T0 = time.monotonic()


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.monotonic() - T0) / 60:4.1f}m] {msg}", flush=True)


def left_s():
    return HARD_STOP_MIN * 60 - (time.monotonic() - T0)


def ssh(ip, cmd, timeout=60, check=False):
    return subprocess.run(["ssh", "-n", "-i", str(KEY), "-o", f"UserKnownHostsFile={KNOWN}", "-o", "StrictHostKeyChecking=accept-new",
                           "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", f"root@{ip}", cmd],
                          capture_output=True, text=True, timeout=timeout, check=check)


def scp(src, dst, timeout=600):
    subprocess.run(["scp", "-q", "-r", "-i", str(KEY), "-o", f"UserKnownHostsFile={KNOWN}", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "BatchMode=yes", *src, dst], check=True, timeout=timeout)


def main():
    load_env_file(str(Path.home() / ".lastscene" / "vultr.env"))
    v = Vultr(os.environ.get("VULTR_API_KEY", ""))
    guard = Guardrails()
    guard.check_date()
    credit = v.credit_left()
    worst = guard.check_cost(PRICE, HARD_STOP_MIN, credit)
    say(f"credit ${credit:.2f}; {PLAN} at ${PRICE:.2f}/h until training finishes (stops if credit < ${CREDIT_FLOOR:.2f})")

    if not KEY.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(KEY), "-C", "lastscene-wake"], check=True)
    KNOWN.write_text("")
    ssh_key_id = v.call("POST", "/ssh-keys", ok=(201, 200), json={"name": "lastscene-wake",
                        "ssh_key": (KEY.with_suffix(".pub")).read_text().strip()})["ssh_key"]["id"]
    instance_id = None
    keep = False  # a failed run keeps the server, so it can be fixed and resumed
    try:
        plans = v.call("GET", "/plans", params={"type": "all", "per_page": 500})["plans"]
        where = next(p for p in plans if p["id"] == PLAN)["locations"]
        region = next(r for r in REGIONS if r in where)
        inst = v.call("POST", "/instances", ok=(202, 201, 200), json={
            "region": region, "plan": PLAN, "os_id": v.ubuntu(), "label": "lastscene-wake", "hostname": "lastscene-wake",
            "tags": ["lastscene-wake"], "sshkey_id": [ssh_key_id], "backups": "disabled", "enable_ipv6": False,
            "activation_email": False})["instance"]
        instance_id = inst["id"]
        say(f"rented {PLAN} in {region}; waiting for it to boot")
        while True:
            inst = v.instance(instance_id)
            if _up(inst) and inst.get("main_ip") not in (None, "", "0.0.0.0"):
                break
            if left_s() < 0:
                raise TimeoutError("hard stop while booting")
            time.sleep(10)
        ip = inst["main_ip"]
        while ssh(ip, "true", timeout=20).returncode != 0:
            if left_s() < 0:
                raise TimeoutError("hard stop waiting for ssh")
            time.sleep(5)
        say("booted; uploading the training scripts")
        ssh(ip, "mkdir -p /root/wake", check=True)
        files = [str(p) for p in (REPO / "device/wake/train").glob("*.py")] + [str(HERE / "remote_train.sh")]
        scp(files, f"root@{ip}:/root/wake/")
        scp([str(REPO / "device/wake/last_scene.onnx")], f"root@{ip}:/root/wake/old_last_scene.onnx")
        # Fully detached (no stdin, own session), so ssh returns at once instead of waiting on the job's pipes.
        # setsid -f forks the job into its own session with every stream redirected, so nothing holds the ssh
        # connection open and the remote shell exits at once. A timeout here only means "go and check".
        try:
            ssh(ip, "cd /root/wake; setsid -f bash remote_train.sh > run.log 2>&1 < /dev/null", timeout=30)
        except subprocess.TimeoutExpired:
            say("start command timed out; checking whether training is running")
        for _ in range(6):
            if ssh(ip, "pgrep -f remote_train.sh >/dev/null && echo yes", timeout=20).stdout.strip() == "yes":
                break
            time.sleep(5)
        else:
            raise RuntimeError("training didn't start on the server")
        say(f"training started on the server ({ip}); no time limit now, only the ${CREDIT_FLOOR:.2f} credit floor")
        shown, checked = 0, time.monotonic()
        while True:
            try:
                out = ssh(ip, "cat /root/wake/run.log", timeout=30).stdout.splitlines()
            except subprocess.TimeoutExpired:  # a slow check is no reason to stop: try again next round
                time.sleep(20)
                continue
            for line in out[shown:]:
                if line.startswith(("==", "--", "torch", "NO GPU", "RIRs", "threshold", "held-out", "near-misses", "lookalikes", "preflight")) \
                        or "start " in line or "Traceback" in line or "Error" in line or "clip length" in line:
                    say("  " + line)
            shown = max(shown, len(out))
            if any(l.startswith("== ") and " DONE" in l for l in out):
                say("training finished")
                break
            if any(l.startswith("== ") and "FAILED" in l for l in out):
                keep = True  # never destroy a failed run: fix it on the server and resume (each stage skips what's done)
                say(f"FAILED on the server; server and ssh key KEPT for a fix: instance {instance_id}, {ip}")
                return
            if time.monotonic() - checked > 300:
                checked = time.monotonic()
                credit = v.credit_left()
                say(f"credit ${credit:.2f}")
                if credit < CREDIT_FLOOR:
                    say("credit floor reached: fetching what exists and stopping")
                    break
            time.sleep(20)
        dest = WORK / "result"
        dest.mkdir(exist_ok=True)
        for f in ("run.log", "out/last_scene.onnx", "out/last_scene.state.pt"):
            try:
                scp([f"root@{ip}:/root/wake/{f}"], str(dest / Path(f).name), timeout=300)
                say(f"fetched {f}")
            except Exception as exc:
                say(f"could not fetch {f}: {exc}")
    finally:
        if keep:
            return
        if instance_id:
            v.destroy(instance_id)
            for _ in range(30):
                if v.instance(instance_id) is None:
                    break
                time.sleep(5)
            say(f"server destroyed ({'gone' if v.instance(instance_id) is None else 'destroy requested'})")
        v.call("DELETE", f"/ssh-keys/{ssh_key_id}", ok=(204, 200, 404))
        say(f"ssh key removed; credit now ${v.credit_left():.2f}")


if __name__ == "__main__":
    main()
