"""Energy metering — the MEASURED denominator for performance-per-watt.

The new north star is intelligence per joule, not just intelligence per step. That
number is worthless if the denominator is an estimate: a post-mortem on an earlier
program burned real time on conclusions that quietly rested on numbers nobody
actually measured. The rule this module exists to enforce is simple and absolute --

    AN ABSENT MEASUREMENT IS NOT A MEASUREMENT OF ZERO.

If the CPU package domain is permission-denied, the CPU term prints as MISSING, not
0.0 -- a 0 there would silently tell every downstream ratio "the CPU used no energy,"
which is a lie an order of magnitude worse than admitting we don't know.

Two independent instruments, two independent failure modes:

  GPU  -- no exact energy counter on this class of card (`power.limit` reports
          [N/A]). We sample `nvidia-smi --query-gpu=power.draw ... -lms <ms>` in a
          background thread and trapezoid-integrate watts over wall-clock receipt
          timestamps (not the nominal interval -- sampling jitter under load makes
          the receipt clock the honest one). This is an ESTIMATE by construction;
          the sampling interval and count travel with every reading so nobody
          mistakes it for a counter.

  CPU  -- intel-rapl exposes an exact hardware energy counter (`energy_uj`), no
          sampling error at all -- when it's readable. On this machine it isn't
          (permission denied); the module must say so plainly and print the exact
          `chmod` line that fixes it, not paper over the gap.

RAPL has its own trap: intel-rapl (MSR) and intel-rapl-mmio (MMIO) expose the SAME
package energy twice, and "psys" (when present) SUPERSETS package rather than
sitting beside it. Summing blindly overcounts by 2x or worse. `rapl_totals` sums
only the intel-rapl:<n> package-level domains once each; psys and subdomains
(core/uncore) are reported individually but never folded into that total.

The GPU idles around 9.7 W on this desktop because Chrome holds it open -- so every
reading here is reported both raw and net-of-idle-baseline; the baseline itself is a
measurement (`idle_baseline`), never a hardcoded constant.

Run:  python -m chronohorn.eval.energy --doctor
      python -m chronohorn.eval.energy --self-test --burn-seconds 6
      python -m chronohorn.eval.energy --wrap --bytes 1000000 -- python -m chronohorn.eval.ngram_control
"""
from __future__ import annotations

import argparse
import functools
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

RAPL_ROOT = Path("/sys/class/powercap")
RAPL_REMEDIATION = (
    "sudo chmod a+r /sys/class/powercap/intel-rapl*/energy_uj "
    "/sys/class/powercap/intel-rapl*/*/energy_uj"
)

# intel-rapl:<pkg> (package/psys) or intel-rapl:<pkg>:<sub> (core/uncore component);
# intel-rapl-mmio mirrors either shape with the SAME package energy over MMIO instead
# of MSR. All of these live as flat siblings directly under /sys/class/powercap --
# the parent/child relationship is encoded in the name, not in directory nesting.
_ID_RE = re.compile(r"^(intel-rapl(?:-mmio)?):(\d+)(?::(\d+))?$")
_PACKAGE_NAME_RE = re.compile(r"^package-\d+$")


# ---------------------------------------------------------------------------
# RAPL: exact hardware energy counters (when readable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaplDomain:
    """One powercap leaf: an MSR package, its psys superset, a package subcomponent
    (core/uncore), or an MMIO mirror of any of these.

    `mirror` is True for anything under intel-rapl-mmio -- discovered (so --doctor
    can show it exists) but always excluded from summation, since it duplicates an
    intel-rapl:* package domain's counter over a different bus.
    """
    id: str                    # e.g. "intel-rapl:0", "intel-rapl:0:0", "intel-rapl-mmio:0"
    name: str                  # contents of `name`: "package-0", "psys", "core", "uncore"
    path: Path
    is_subdomain: bool         # True for intel-rapl:<pkg>:<sub> (a component of a package)
    parent_id: str | None      # owning package's id, for subdomains only
    mirror: bool               # True under intel-rapl-mmio (MSR duplicate over MMIO)
    readable: bool             # both energy_uj and max_energy_range_uj readable right now
    max_range_uj: int | None   # None unless readable

    def read_uj(self) -> int:
        """Current raw counter value. Raises OSError if not actually readable."""
        return int((self.path / "energy_uj").read_text().strip())


def _try_read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def discover_rapl_domains(root: Path = RAPL_ROOT) -> list[RaplDomain]:
    """Classify every intel-rapl* leaf under `root`.

    Pure filesystem walk (a `name` read plus a permission probe on energy_uj /
    max_energy_range_uj) -- side-effect-free, safe for --doctor. `root` is a
    parameter, not a hardcoded path, precisely so tests can point it at a fake
    tree under tmp_path instead of the real /sys/class/powercap.
    """
    domains: list[RaplDomain] = []
    if not root.exists():
        return domains
    for entry in sorted(root.iterdir()):
        m = _ID_RE.match(entry.name)
        if not m:
            continue
        base, pkg, sub = m.group(1), m.group(2), m.group(3)
        mirror = base.endswith("-mmio")
        is_subdomain = sub is not None
        parent_id = f"{base}:{pkg}" if is_subdomain else None
        try:
            name = (entry / "name").read_text().strip()
        except OSError:
            name = entry.name
        energy_uj = _try_read_int(entry / "energy_uj")
        max_range = _try_read_int(entry / "max_energy_range_uj")
        readable = energy_uj is not None and max_range is not None
        domains.append(RaplDomain(
            id=entry.name, name=name, path=entry, is_subdomain=is_subdomain,
            parent_id=parent_id, mirror=mirror, readable=readable, max_range_uj=max_range,
        ))
    return domains


def rapl_delta_uj(start_uj: int, end_uj: int, max_range_uj: int) -> int:
    """Counter delta with wraparound handling.

    RAPL counters wrap at `max_energy_range_uj` and restart at 0. A negative raw
    delta means exactly one wrap happened in the interval (these counters are read
    far more often than they can wrap twice); add the range back to recover the
    true delta.
    """
    delta = end_uj - start_uj
    if delta < 0:
        delta += max_range_uj
    return delta


def is_package_domain(name: str) -> bool:
    """True for a top-level MSR package name ("package-0", "package-1", ...).

    Deliberately excludes "psys" (a superset of package, never summed with it) and
    subdomain names like "core"/"uncore" (components of a package, not sockets).
    """
    return bool(_PACKAGE_NAME_RE.match(name))


def rapl_totals(domains: list[RaplDomain], deltas_j: dict[str, float]) -> dict[str, float]:
    """Per-domain joules keyed by human name, ready for `EnergyReading.cpu_domains`.

    Top-level domains (package-N, psys) keep their bare name; subdomains are
    namespaced under their parent ("package-0/core") so they can never collide
    with, or be mistaken for, a package total by `is_package_domain`. Mirror
    (intel-rapl-mmio*) domains are dropped entirely -- see module docstring on
    double counting.
    """
    by_id = {d.id: d for d in domains}
    by_name: dict[str, float] = {}
    for d in domains:
        if d.mirror or d.id not in deltas_j:
            continue
        if d.is_subdomain and d.parent_id in by_id:
            key = f"{by_id[d.parent_id].name}/{d.name}"
        else:
            key = d.name
        by_name[key] = by_name.get(key, 0.0) + deltas_j[d.id]
    return by_name


def rapl_package_total(by_name: dict[str, float]) -> float:
    """Sum of package-level domains only -- excludes psys and subdomains."""
    return sum(v for k, v in by_name.items() if is_package_domain(k))


# ---------------------------------------------------------------------------
# GPU: sampled power, trapezoid-integrated (an estimate, never a counter)
# ---------------------------------------------------------------------------


def parse_power_draw_line(line: str) -> float | None:
    """Parse one `nvidia-smi --query-gpu=power.draw` CSV line, or None if it isn't
    a number (GPU asleep, driver hiccup: "N/A", "[N/A]", blank, garbage). Callers
    must count these, never treat them as a 0 W sample.
    """
    s = line.strip().strip("[]")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def trapezoid_integrate_j(timestamps: list[float], watts: list[float]) -> float:
    """Trapezoidal-rule integral of power (W) over wall-clock timestamps (s) -> J.

    Fewer than two samples can't be integrated -- returns 0.0 (the caller is
    responsible for treating "not enough samples" as MISSING, not as "0 joules
    used", when that matters).
    """
    if len(timestamps) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        total += 0.5 * (watts[i] + watts[i - 1]) * dt
    return total


@dataclass
class GpuPowerSamples:
    timestamps: list[float]
    watts: list[float]
    na_count: int
    available: bool     # False only if nvidia-smi itself could not be launched

    @property
    def joules(self) -> float | None:
        """None (not 0.0) when there weren't enough usable samples to integrate."""
        if not self.available or len(self.timestamps) < 2:
            return None
        return trapezoid_integrate_j(self.timestamps, self.watts)

    @property
    def avg_w(self) -> float | None:
        if not self.available or not self.watts:
            return None
        return sum(self.watts) / len(self.watts)


class GpuPowerSampler:
    """Background `nvidia-smi -lms <interval_ms>` power sampler.

    Each line is stamped with a wall-clock receipt timestamp (`time.monotonic()`),
    not its nominal interval time -- sampling jitter under load makes the receipt
    clock the more honest one for trapezoid integration. Unparseable / "N/A" lines
    are skipped and counted, never folded in as 0 W.
    """

    def __init__(self, interval_ms: int = 200, nvidia_smi: str = "nvidia-smi") -> None:
        self.interval_ms = interval_ms
        self.nvidia_smi = nvidia_smi
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._timestamps: list[float] = []
        self._watts: list[float] = []
        self._na_count = 0
        self._available = True

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [self.nvidia_smi, "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                 "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except FileNotFoundError:
            self._available = False
            return
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, ""):
            ts = time.monotonic()
            watt = parse_power_draw_line(line)
            with self._lock:
                if watt is None:
                    self._na_count += 1
                else:
                    self._timestamps.append(ts)
                    self._watts.append(watt)

    def stop(self) -> GpuPowerSamples:
        """Terminate the subprocess and join the reader thread -- always, even if
        the metered block above raised. Safe to call even if start() never ran or
        nvidia-smi was never found.
        """
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=2.0)
                except Exception:
                    pass
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            ts, watts, na = list(self._timestamps), list(self._watts), self._na_count
        return GpuPowerSamples(timestamps=ts, watts=watts, na_count=na, available=self._available)


def _gpu_power_once(nvidia_smi: str = "nvidia-smi", timeout: float = 5.0) -> float | None:
    """One-shot instantaneous power.draw read -- for --doctor, not for metering."""
    try:
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return parse_power_draw_line(out.stdout.strip().splitlines()[0])


# ---------------------------------------------------------------------------
# EnergyReading / EnergyMeter
# ---------------------------------------------------------------------------


@dataclass
class EnergyReading:
    """One metered window's energy, per domain, raw and baseline-net.

    `cpu_domains` is empty (never zero-filled) when RAPL wasn't readable; `missing`
    then names exactly what's absent and how to fix it. `net_gpu_j` / `net_cpu_j`
    are None whenever there is no baseline to subtract -- "no baseline" is not the
    same claim as "net energy is equal to raw energy", so it is left unstated
    rather than guessed.
    """
    duration_s: float
    gpu_j: float | None
    gpu_avg_w: float | None
    gpu_samples: int
    cpu_domains: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    baseline_gpu_w: float | None = None
    baseline_cpu_w: float | None = None

    @property
    def cpu_j(self) -> float | None:
        """Package-level CPU total, or None if no CPU domain was measured."""
        if not self.cpu_domains:
            return None
        return rapl_package_total(self.cpu_domains)

    @property
    def net_gpu_j(self) -> float | None:
        if self.gpu_j is None or self.baseline_gpu_w is None:
            return None
        return max(0.0, self.gpu_j - self.baseline_gpu_w * self.duration_s)

    @property
    def net_cpu_j(self) -> float | None:
        cj = self.cpu_j
        if cj is None or self.baseline_cpu_w is None:
            return None
        return max(0.0, cj - self.baseline_cpu_w * self.duration_s)


def format_energy_line(reading: EnergyReading, n_bytes: int | None = None) -> str:
    """The one parseable log line other tools grep for. MISSING is a literal token,
    never rendered as 0 or 0.0 -- that distinction is the entire point of this
    module.
    """
    def fmt(x: float | None, digits: int = 1) -> str:
        return "MISSING" if x is None else f"{x:.{digits}f}"

    parts = [
        f"dur_s={reading.duration_s:.2f}",
        f"gpu_J={fmt(reading.gpu_j)}",
        f"gpu_avg_W={fmt(reading.gpu_avg_w)}",
        f"gpu_net_J={fmt(reading.net_gpu_j)}",
        f"cpu_J={fmt(reading.cpu_j)}",
        f"samples={reading.gpu_samples}",
    ]
    if n_bytes is not None:
        numer = reading.net_gpu_j if reading.baseline_gpu_w is not None else reading.gpu_j
        if numer is None or n_bytes <= 0:
            parts.append("J_per_byte=MISSING")
        else:
            parts.append(f"J_per_byte={numer / n_bytes:.3g}")
    if reading.missing:
        parts.append("missing=[" + "; ".join(reading.missing) + "]")
    return "ENERGY: " + " ".join(parts)


class EnergyMeter:
    """`with EnergyMeter(interval_ms=200, baseline=idle) as m: ...` then `m.reading`.

    Snapshots readable, non-mirror RAPL domains and starts the GPU sampler on
    entry; on exit (success OR exception -- the context-manager protocol
    guarantees __exit__ runs either way) it always stops the sampler first
    (terminate + join, never leaked) before taking the closing RAPL snapshot.
    """

    def __init__(self, interval_ms: int = 200, baseline: "IdleBaseline | None" = None,
                 rapl_root: Path = RAPL_ROOT, nvidia_smi: str = "nvidia-smi") -> None:
        self.interval_ms = interval_ms
        self.baseline = baseline
        self.rapl_root = rapl_root
        self.nvidia_smi = nvidia_smi
        self.reading: EnergyReading | None = None
        self._sampler: GpuPowerSampler | None = None
        self._rapl_domains: list[RaplDomain] = []
        self._rapl_start: dict[str, int] = {}
        self._enter_missing: list[str] = []
        self._t0 = 0.0

    def __enter__(self) -> "EnergyMeter":
        all_domains = discover_rapl_domains(self.rapl_root)
        self._rapl_domains = [d for d in all_domains if d.readable and not d.mirror]
        self._enter_missing = []
        if not self._rapl_domains:
            self._enter_missing.append(f"cpu-rapl: permission denied — run: {RAPL_REMEDIATION}")
        try:
            self._rapl_start = {d.id: d.read_uj() for d in self._rapl_domains}
        except OSError as exc:
            self._rapl_start = {}
            self._enter_missing.append(f"cpu-rapl: read failed ({exc}) — run: {RAPL_REMEDIATION}")

        self._sampler = GpuPowerSampler(self.interval_ms, self.nvidia_smi)
        self._sampler.start()
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = time.monotonic() - self._t0
        try:
            gpu_samples = self._sampler.stop() if self._sampler is not None else None
        except Exception:
            gpu_samples = None

        missing = list(self._enter_missing)
        cpu_domains: dict[str, float] = {}
        if self._rapl_domains and self._rapl_start:
            try:
                end = {d.id: d.read_uj() for d in self._rapl_domains}
            except OSError as read_exc:
                missing.append(f"cpu-rapl: read failed mid-run ({read_exc}) — run: {RAPL_REMEDIATION}")
            else:
                deltas_j = {
                    d.id: rapl_delta_uj(self._rapl_start[d.id], end[d.id], d.max_range_uj) / 1e6
                    for d in self._rapl_domains
                    if d.id in self._rapl_start and d.id in end
                }
                cpu_domains = rapl_totals(self._rapl_domains, deltas_j)

        if gpu_samples is None or not gpu_samples.available:
            missing.append(f"gpu: '{self.nvidia_smi}' not available")
            gpu_j = gpu_avg_w = None
            n_samples = 0
        else:
            n_samples = len(gpu_samples.watts)
            gpu_j = gpu_samples.joules
            gpu_avg_w = gpu_samples.avg_w
            if gpu_j is None:
                missing.append(f"gpu: only {n_samples} usable power sample(s) — need >=2 to integrate")

        baseline_gpu_w = self.baseline.gpu_w if self.baseline is not None else None
        baseline_cpu_w = (
            rapl_package_total(self.baseline.cpu_w)
            if self.baseline is not None and self.baseline.cpu_w else None
        )

        self.reading = EnergyReading(
            duration_s=duration, gpu_j=gpu_j, gpu_avg_w=gpu_avg_w, gpu_samples=n_samples,
            cpu_domains=cpu_domains, missing=missing,
            baseline_gpu_w=baseline_gpu_w, baseline_cpu_w=baseline_cpu_w,
        )
        return None


@dataclass
class IdleBaseline:
    """Average idle power per available domain, MEASURED immediately before a run
    -- never a hardcoded constant. On this desktop the GPU idles around 9.7 W
    because Chrome holds it open; without subtracting a fresh baseline, every
    reading double-counts that floor as if the workload produced it.
    """
    seconds: float
    gpu_w: float | None
    cpu_w: dict[str, float]
    missing: list[str]


def idle_baseline(seconds: float = 8.0, gpu_interval_ms: int = 200,
                   rapl_root: Path = RAPL_ROOT, nvidia_smi: str = "nvidia-smi") -> IdleBaseline:
    """Measure avg idle watts per domain over `seconds` of doing nothing."""
    with EnergyMeter(interval_ms=gpu_interval_ms, rapl_root=rapl_root, nvidia_smi=nvidia_smi) as m:
        time.sleep(seconds)
    r = m.reading
    assert r is not None
    cpu_w = {k: v / r.duration_s for k, v in r.cpu_domains.items()} if r.cpu_domains else {}
    return IdleBaseline(seconds=r.duration_s, gpu_w=r.gpu_avg_w, cpu_w=cpu_w, missing=r.missing)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt_missing(x: float | None, unit: str = "") -> str:
    return "MISSING" if x is None else f"{x:.3f}{unit}"


def doctor_lines(rapl_root: Path = RAPL_ROOT, nvidia_smi: str = "nvidia-smi") -> list[str]:
    """What's measurable on this machine right now. No side effects beyond reading
    sysfs metadata and one one-shot nvidia-smi query (not a persistent process).
    """
    lines = ["ENERGY DOCTOR — what is measurable on this machine right now", ""]

    domains = discover_rapl_domains(rapl_root)
    if not domains:
        lines.append(f"cpu-rapl: no intel-rapl domains found under {rapl_root}")
    else:
        lines.append("cpu-rapl domains:")
        any_denied = False
        for d in domains:
            role = "mirror(mmio)" if d.mirror else ("subdomain of " + str(d.parent_id) if d.is_subdomain else "package-level")
            status = "READABLE" if d.readable else "PERMISSION DENIED"
            any_denied = any_denied or not d.readable
            lines.append(f"  {d.id:<20} name={d.name:<10} {role:<22} {status}")
        if any_denied:
            lines.append("")
            lines.append(f"  remediation: {RAPL_REMEDIATION}")
    lines.append("")

    which = shutil.which(nvidia_smi)
    if which is None:
        lines.append(f"gpu: '{nvidia_smi}' not found on PATH — GPU domain will read MISSING")
    else:
        w = _gpu_power_once(nvidia_smi)
        if w is None:
            lines.append(f"gpu: '{nvidia_smi}' found at {which} but power.draw query failed/unsupported")
        else:
            lines.append(f"gpu: nvidia-smi power.draw WORKING — current draw {w:.2f} W (instantaneous)")
    return lines


def _cpu_burn(seconds: float) -> None:
    """Busy CPU work for ~`seconds`: repeated numpy matmuls. Not tuned for FLOPs --
    just enough sustained load to produce a non-trivial package power draw.
    """
    import numpy as np

    end = time.monotonic() + seconds
    n = 512
    a = np.random.default_rng(0).standard_normal((n, n))
    b = np.random.default_rng(1).standard_normal((n, n))
    while time.monotonic() < end:
        a = (a @ b) * 1e-6 + a
        b = (b @ a) * 1e-6 + b


def _gpu_burn(seconds: float, device: str = "cuda") -> bool:
    """Busy GPU work for ~`seconds`: repeated torch matmuls, synced at the boundary
    (CUDA is async by default -- without the sync the metering window would close
    before the work actually finished). Returns False (SKIPPED) if torch or CUDA
    aren't available -- imported lazily so this module has no hard torch dependency.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    n = 2048
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        a = (a @ b) * 1e-6 + a
        b = (b @ a) * 1e-6 + b
    torch.cuda.synchronize()
    return True


def _self_test(log, burn_seconds: float = 6.0, interval_ms: int = 200) -> int:
    log("=" * 70)
    log("DOCTOR")
    log("=" * 70)
    for line in doctor_lines():
        log(line)

    log("")
    log("idle baseline (measuring, not assuming)...")
    baseline = idle_baseline(seconds=3.0, gpu_interval_ms=interval_ms)
    log(f"  gpu idle:            {_fmt_missing(baseline.gpu_w, ' W')}")
    cpu_idle = rapl_package_total(baseline.cpu_w) if baseline.cpu_w else None
    log(f"  cpu idle (package):  {_fmt_missing(cpu_idle, ' W')}")
    if baseline.missing:
        for m in baseline.missing:
            log(f"  missing: {m}")

    rows: list[tuple[str, EnergyReading]] = []

    log("")
    log(f"CPU burn ({burn_seconds:.0f}s, numpy matmul loop)...")
    with EnergyMeter(interval_ms=interval_ms, baseline=baseline) as m:
        _cpu_burn(burn_seconds)
    assert m.reading is not None
    rows.append(("cpu-burn", m.reading))
    log(format_energy_line(m.reading))

    log("")
    log(f"GPU burn ({burn_seconds:.0f}s, torch matmul loop on cuda)...")
    with EnergyMeter(interval_ms=interval_ms, baseline=baseline) as m2:
        ran = _gpu_burn(burn_seconds)
    if not ran:
        log("GPU burn: SKIPPED (torch or CUDA unavailable)")
    else:
        assert m2.reading is not None
        rows.append(("gpu-burn", m2.reading))
        log(format_energy_line(m2.reading))

    log("")
    log("=" * 70)
    log("SELF-TEST SUMMARY (raw / net-of-idle-baseline, joules)")
    log("=" * 70)
    log(f"{'burn':<10} {'gpu_J':>10} {'gpu_net_J':>10} {'cpu_J':>10} {'cpu_net_J':>10}")
    for name, r in rows:
        log(f"{name:<10} {_fmt_missing(r.gpu_j):>10} {_fmt_missing(r.net_gpu_j):>10} "
            f"{_fmt_missing(r.cpu_j):>10} {_fmt_missing(r.net_cpu_j):>10}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doctor", action="store_true",
                    help="report what's measurable on this machine right now; no side effects")
    ap.add_argument("--self-test", action="store_true",
                    help="doctor + idle baseline + metered CPU/GPU burns, end to end")
    ap.add_argument("--burn-seconds", type=float, default=6.0)
    ap.add_argument("--wrap", action="store_true",
                    help="meter an arbitrary child command: --wrap [opts] -- <cmd...>")
    ap.add_argument("--bytes", dest="n_bytes", type=int, default=None,
                    help="byte count for J/byte, used with --wrap")
    ap.add_argument("--baseline-seconds", type=float, default=None,
                    help="take an idle baseline of this many seconds before --wrap")
    ap.add_argument("--interval-ms", type=int, default=200)
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="with --wrap: the child command, after a literal --")
    a = ap.parse_args(argv)
    log = functools.partial(print, flush=True)

    if a.doctor:
        for line in doctor_lines():
            log(line)
        return 0

    if a.self_test:
        return _self_test(log, burn_seconds=a.burn_seconds, interval_ms=a.interval_ms)

    if a.wrap:
        cmd = list(a.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            log("ERROR: --wrap requires a command after --, e.g. --wrap -- python foo.py")
            return 2
        baseline = idle_baseline(a.baseline_seconds) if a.baseline_seconds else None
        with EnergyMeter(interval_ms=a.interval_ms, baseline=baseline) as m:
            proc = subprocess.run(cmd)
        assert m.reading is not None
        log(format_energy_line(m.reading, n_bytes=a.n_bytes))
        return proc.returncode

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
