"""Background GPU telemetry via nvidia-smi (utilization, power, temperature, clock).

Does nothing (and says so) if nvidia-smi isn't available. Usage:
    with GpuMonitor(log_path="data/gpu_log.csv") as gpu:
        ...work...
    print(gpu.summary())
"""

import csv
import shutil
import subprocess
import threading
import time

FIELDS = ["utilization.gpu", "power.draw", "temperature.gpu", "clocks.sm", "memory.used"]
NAMES = ["util_pct", "power_w", "temp_c", "sm_clock_mhz", "mem_used_mib"]


class GpuMonitor:
    def __init__(self, interval_s: float = 1.0, log_path: str | None = None):
        self.interval_s = interval_s
        self.log_path = log_path
        self.samples: list[dict] = []
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.log_path and self.samples:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["t_s", *NAMES])
                writer.writeheader()
                writer.writerows(self.samples)

    def _run(self):
        start = time.monotonic()
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                values = [float(v) if v.strip() not in ("[N/A]", "N/A") else None for v in out.split(",")]
                self.samples.append({"t_s": round(time.monotonic() - start, 1), **dict(zip(NAMES, values))})
            except (subprocess.SubprocessError, IndexError, ValueError):
                pass
            self._stop.wait(self.interval_s)

    def summary(self) -> dict:
        if not self.available:
            return {"gpu": "nvidia-smi not available"}
        if not self.samples:
            return {"gpu": "no samples"}

        def col(name):
            return [s[name] for s in self.samples if s[name] is not None]

        out = {"samples": len(self.samples)}
        for name in ("util_pct", "power_w", "temp_c"):
            values = col(name)
            if values:
                out[f"{name}_mean"] = round(sum(values) / len(values), 1)
                out[f"{name}_max"] = round(max(values), 1)
        return out
