"""Jetson `tegrastats` power / thermal sampling for edge LNN benchmarks.

Why this exists
---------------
Every Jetson report in `analysis/jetson/` so far reports accuracy, parameter
count, throughput and train time -- but **no energy**. On an edge device the
question is rarely "which model is fastest" but "which model gives the best
accuracy per joule inside a 7W / 15W / 25W power envelope". A CfC that is 5x
slower than a GRU but draws 1/3 the power can still be the right answer.

This module wraps `tegrastats` (present on every JetPack install, no extra
deps) in a background sampler so any benchmark can do::

    with TegrastatsSampler() as sampler:
        run_model(...)
    energy = sampler.summary()

and get mean/peak power per rail, temperatures, GPU utilisation and the
integrated energy (mJ) over the sampled window.

Rails on Orin Nano Super (R36.4.x)
----------------------------------
- ``VDD_IN``            total board input power -- the number that matters for
                        a battery-powered / PoE edge deployment.
- ``VDD_CPU_GPU_CV``    CPU + GPU + DLA/PVA compute rail -- the number that
                        isolates *model* cost from board overhead.
- ``VDD_SOC``           SoC fabric, memory controller, IO.

Note that older Jetsons (Nano, TX2, Xavier) expose different rail names
(``POM_5V_IN``, ``VDD_GPU_SOC``, ...). The parser is rail-name agnostic: it
picks up every ``NAME cur/avg`` token it sees, so the same code works across
generations. Boards with no INA3221 (e.g. some Orin NX modules) simply yield
an empty ``power`` dict and the caller degrades gracefully.

Safety
------
This is a **read-only observer**. It spawns `tegrastats` and parses stdout.
It never writes to sysfs, never changes power mode / clocks, never touches
`jetson_clocks` or `nvpmodel`. Nothing here can alter device state.
"""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any

__all__ = [
    "TegrastatsSampler",
    "parse_tegrastats_line",
    "tegrastats_available",
]


# `VDD_IN 6000mW/6000mW` -> ("VDD_IN", 6000, 6000)
_POWER_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+(\d+)mW/(\d+)mW")
# `gpu@50.718C` / `tj@50.718C`
_TEMP_RE = re.compile(r"\b([a-zA-Z0-9_]+)@(-?\d+(?:\.\d+)?)C")
# `GR3D_FREQ 0%` (Orin) or `GR3D_FREQ 0%@[306]` (older BSPs)
_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
# `RAM 5801/7620MB`
_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
# `CPU [37%@729,22%@729,...]`
_CPU_RE = re.compile(r"CPU\s+\[([^\]]*)\]")
_CPU_CORE_RE = re.compile(r"(\d+)%@(\d+)")


def tegrastats_available() -> bool:
    """True when the `tegrastats` binary is on PATH (i.e. we are on a Jetson)."""
    return shutil.which("tegrastats") is not None


def parse_tegrastats_line(line: str) -> dict[str, Any]:
    """Parse one `tegrastats` stdout line into a flat sample dict.

    Unknown / missing fields are simply absent from the result rather than
    filled with sentinels, so a caller can distinguish "board has no INA3221"
    from "rail read 0 mW".
    """
    sample: dict[str, Any] = {}

    power = {name: cur for name, cur, _avg in _POWER_RE.findall(line)}
    if power:
        sample["power_mw"] = {name: int(value) for name, value in power.items()}

    temps = {name: float(value) for name, value in _TEMP_RE.findall(line)}
    if temps:
        sample["temp_c"] = temps

    gr3d = _GR3D_RE.search(line)
    if gr3d:
        sample["gpu_util_pct"] = int(gr3d.group(1))

    ram = _RAM_RE.search(line)
    if ram:
        sample["ram_used_mb"] = int(ram.group(1))
        sample["ram_total_mb"] = int(ram.group(2))

    cpu = _CPU_RE.search(line)
    if cpu:
        cores = _CPU_CORE_RE.findall(cpu.group(1))
        if cores:
            sample["cpu_util_pct"] = [int(util) for util, _freq in cores]
            sample["cpu_freq_mhz"] = [int(freq) for _util, freq in cores]

    return sample


class TegrastatsSampler:
    """Background `tegrastats` sampler usable as a context manager.

    Args:
        interval_ms: tegrastats sampling period. 100 ms is the practical floor;
            below that the parser cost starts to show up in the measurement.
        warmup_s: seconds to wait after starting tegrastats before counting
            samples, so the first (often stale) line does not skew the mean.

    Example::

        with TegrastatsSampler(interval_ms=100) as sampler:
            model(x)
        print(sampler.summary()["power_mw"]["VDD_IN"]["mean"])

    If tegrastats is unavailable the sampler degrades to a no-op and
    ``summary()`` returns ``{"available": False, ...}``. Callers therefore
    never need to branch on platform.
    """

    def __init__(self, interval_ms: int = 100, warmup_s: float = 0.0) -> None:
        self.interval_ms = max(int(interval_ms), 50)
        self.warmup_s = max(float(warmup_s), 0.0)
        self.samples: list[dict[str, Any]] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._error: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "TegrastatsSampler":
        if not tegrastats_available():
            self._error = "tegrastats binary not found (not a Jetson, or not on PATH)"
            return self
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._error = f"failed to spawn tegrastats: {exc}"
            return self

        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        if self.warmup_s:
            time.sleep(self.warmup_s)
        # Discard whatever landed during warmup so the window is clean.
        self.samples.clear()
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> "TegrastatsSampler":
        self._stop_time = time.perf_counter()
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        return self

    def __enter__(self) -> "TegrastatsSampler":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- internals ---------------------------------------------------------

    def _reader(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self._stop.is_set():
                break
            sample = parse_tegrastats_line(line)
            if sample:
                self.samples.append(sample)

    # -- reporting ---------------------------------------------------------

    @property
    def duration_s(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.perf_counter()
        return max(end - self._start_time, 0.0)

    def summary(self) -> dict[str, Any]:
        """Aggregate the sampled window.

        Returns a dict with:
            available:     whether any sample was collected
            n_samples:     number of parsed tegrastats lines
            duration_s:    wall-clock length of the sampled window
            power_mw:      {rail: {mean, peak, min}} for every rail seen
            energy_mj:     {rail: mean_mW * duration_s} integrated energy
            temp_c:        {zone: {mean, peak}} for every thermal zone seen
            gpu_util_pct:  {mean, peak}
            ram_used_mb:   {mean, peak}
        """
        if self._error is not None and not self.samples:
            return {"available": False, "reason": self._error, "n_samples": 0}
        if not self.samples:
            return {
                "available": False,
                "reason": "tegrastats produced no parseable samples "
                "(window too short? try a longer run or smaller --interval)",
                "n_samples": 0,
            }

        duration = self.duration_s
        out: dict[str, Any] = {
            "available": True,
            "n_samples": len(self.samples),
            "duration_s": round(duration, 4),
            "interval_ms": self.interval_ms,
        }

        # Power rails -> mean/peak/min + integrated energy.
        rails: dict[str, list[int]] = {}
        for sample in self.samples:
            for rail, value in sample.get("power_mw", {}).items():
                rails.setdefault(rail, []).append(value)
        if rails:
            out["power_mw"] = {
                rail: {
                    "mean": round(statistics.fmean(values), 1),
                    "peak": max(values),
                    "min": min(values),
                }
                for rail, values in rails.items()
            }
            # mW * s = mJ. This is the headline edge metric.
            out["energy_mj"] = {
                rail: round(statistics.fmean(values) * duration, 1)
                for rail, values in rails.items()
            }

        zones: dict[str, list[float]] = {}
        for sample in self.samples:
            for zone, value in sample.get("temp_c", {}).items():
                zones.setdefault(zone, []).append(value)
        if zones:
            out["temp_c"] = {
                zone: {"mean": round(statistics.fmean(values), 2), "peak": max(values)}
                for zone, values in zones.items()
            }

        gpu = [s["gpu_util_pct"] for s in self.samples if "gpu_util_pct" in s]
        if gpu:
            out["gpu_util_pct"] = {"mean": round(statistics.fmean(gpu), 1), "peak": max(gpu)}

        ram = [s["ram_used_mb"] for s in self.samples if "ram_used_mb" in s]
        if ram:
            out["ram_used_mb"] = {"mean": round(statistics.fmean(ram), 1), "peak": max(ram)}

        return out


def energy_per_step(summary: dict[str, Any], steps: int, rail: str = "VDD_IN") -> float | None:
    """mJ consumed per inference step, the accuracy-per-joule denominator.

    Returns None when the rail was not sampled (non-Jetson, or no INA3221).
    """
    if not summary.get("available") or steps <= 0:
        return None
    energy = summary.get("energy_mj", {}).get(rail)
    if energy is None:
        return None
    return energy / steps
