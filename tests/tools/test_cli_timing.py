"""CLI timing honesty gate (kernel-hardening II).

The headline `Total` was a SUM OF KERNEL TIMERS, not wall time — it under-reported
the real run by up to ~13x and made `Realtime` a fiction. The fix (spectral_cli_pipeline.c)
makes `Total` the real monotonic wall span and derives `Realtime` from it, with a Busy/Idle
breakdown. This test pins that honesty against the INDEPENDENT stderr stage markers:

  - Total (wall) >= Busy (sum of per-stage kernel timers)         [wall can't be < kernel]
  - Total ~>= the stage-marker wall (first BEGIN .. last END)      [Total IS wall, not the sum]
  - Realtime == audio_dur / (Total/1000)                          [realtime derived from wall]

Fail-on-bug: revert Total to the kernel sum and the marker-wall assertion fails (the sum is
strictly below the marker wall, which contains alloc/setup the kernel timers don't).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from spectral_tools.core.constants import STAGE_MARKER_LINE_RE  # noqa: E402 — canonical marker regex

WAV = ROOT / "resources/testing/shakespeare_he_saw_the_cat.wav"
BENCH_ARGS = ["0", "1.0", "0", "4096", "128", "-90", "8", "1"]

FFT_LINE = re.compile(
    r"FFT:\s*([\d.]+)ms\s*Track:\s*([\d.]+)ms\s*Synth:\s*([\d.]+)ms\s*"
    r"Norm:\s*([\d.]+)ms\s*Write:\s*([\d.]+)ms\s*Total:\s*([\d.]+)ms")
BUSY_LINE = re.compile(r"Busy:\s*([\d.]+)ms\s*Idle:\s*([\d.]+)ms")
RT_LINE = re.compile(r"Audio:\s*([\d.]+)s\s*Realtime:\s*([\d.]+)x")
STAGE_IDLE_LINE = re.compile(r"Stage idle:.*synth\s+([\d.]+)ms")


@pytest.fixture(scope="session")
def desktop_binary(tmp_path_factory) -> Path:
    """Build desktop once; return the binary path (shared by the run-based tests)."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not on PATH")
    if not WAV.exists():
        pytest.skip(f"bench fixture missing: {WAV}")
    build = tmp_path_factory.mktemp("cli_timing")
    cfg = subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build),
                          "-DCMAKE_BUILD_TYPE=Release"], capture_output=True, text=True)
    if cfg.returncode != 0:
        pytest.skip(f"configure failed:\n{cfg.stderr[-1500:]}")
    bld = subprocess.run(["cmake", "--build", str(build), "--target", "desktop", "-j8"],
                         capture_output=True, text=True)
    if bld.returncode != 0:
        pytest.skip(f"desktop build failed:\n{bld.stderr[-1500:]}")
    bins = list((build / "bin").glob("spectral_*desktop"))
    if not bins:
        pytest.skip("no desktop binary produced")
    return bins[0]


@pytest.fixture(scope="session")
def cli_run(desktop_binary) -> str:
    """Run the CLI on the bench fixture (non-cache); return stdout+stderr."""
    run = subprocess.run([str(desktop_binary), str(WAV), *BENCH_ARGS], capture_output=True, text=True)
    assert run.returncode == 0, f"CLI run failed: {run.stderr[-1500:]}"
    return run.stdout + run.stderr


def _parse(out: str):
    fft = FFT_LINE.search(out)
    busy = BUSY_LINE.search(out)
    rt = RT_LINE.search(out)
    assert fft and busy and rt, "could not parse the Timing section (format changed?)"
    stages = [float(fft.group(i)) for i in range(1, 6)]
    total = float(fft.group(6))
    # canonical stage-marker protocol: ^SPECTRAL_STAGE_(BEGIN|END) <name> <ns>$ -> (kind, ns)
    markers = []
    for line in out.splitlines():
        mm = STAGE_MARKER_LINE_RE.match(line.strip())
        if mm:
            markers.append((mm.group(1), int(mm.group(3))))
    return stages, total, float(busy.group(1)), float(busy.group(2)), \
        float(rt.group(1)), float(rt.group(2)), markers


def test_total_is_wall_not_kernel_sum(cli_run):
    stages, total, busy, idle, _audio, _rt, markers = _parse(cli_run)
    assert abs(busy - sum(stages)) <= 0.6, f"Busy {busy} != sum of stages {sum(stages)}"
    assert total >= busy - 0.6, f"Total(wall) {total} < Busy(kernel sum) {busy} — impossible"
    assert abs((total - busy) - idle) <= 0.6, f"Idle {idle} != Total-Busy {total-busy}"
    if markers:
        begins = [ns for k, ns in markers if k == "BEGIN"]
        ends = [ns for k, ns in markers if k == "END"]
        marker_wall_ms = (max(ends) - min(begins)) / 1e6
        # Total is the WALL span: it must reach the independently-measured marker wall.
        # The kernel sum (the old bug) is strictly below it. 5% slack for clock granularity.
        assert total >= marker_wall_ms * 0.95, (
            f"Total {total}ms < marker wall {marker_wall_ms:.1f}ms*0.95 — Total looks like the "
            f"kernel sum ({busy}ms), not wall (plan §II regression)")


def test_realtime_is_derived_from_wall(cli_run):
    _stages, total, _busy, _idle, audio, realtime, _markers = _parse(cli_run)
    expect = audio / (total / 1000.0)
    assert abs(realtime - expect) / expect < 0.02, (
        f"Realtime {realtime}x != audio/wall {expect:.1f}x — realtime not derived from the "
        f"wall Total (plan §II)")


def test_per_stage_idle_present(cli_run):
    """II.3: a per-stage idle line attributes each stage's wall-minus-kernel hidden time
    (synth idle = backend init/dispatch overhead). Fail-on-bug: drop the per-stage wall
    capture and the line vanishes."""
    m = STAGE_IDLE_LINE.search(cli_run)
    assert m, "the per-stage 'Stage idle:' line is missing (plan §II.3)"
    assert float(m.group(1)) >= 0.0, f"synth idle must be non-negative, got {m.group(1)}"


def test_analysis_path_decision_logged(cli_run):
    """V (logging): the analysis path decision (full_matrix vs spsc_pipeline) is a major
    capability decision and must be visible in the output. Fail-on-bug: neuter the
    'Analysis crossover' log and this fails."""
    assert re.search(r"Analysis crossover:.*path=(full_matrix|spsc_pipeline)", cli_run), (
        "the analysis path decision must be logged (plan §V / AI_CANON #29)")


def test_memory_report_is_honest(cli_run):
    """II.4: the old 'Peak tracked' line summed a near-unwired instrumented subset (~3% of
    RSS) — a naming lie. It must be GONE, replaced by the real RSS delta, plus a Faults line
    surfacing page faults / context switches / realloc count."""
    assert "Peak tracked" not in cli_run, (
        "the misleading 'Peak tracked' memory figure is back (plan §II.4)")
    assert re.search(r"Memory:\s*RSS\s+\d+\s*MB\s*\([+-]\d+\s*MB", cli_run), (
        "Memory line must report RSS + the real resident delta (+/- MB this run)")
    assert re.search(r"Faults:\s*major\s+\d+.*reallocs\s+\d+", cli_run), (
        "Faults line must report major faults, ctx-switches, and the realloc count")


def test_paths_record_reflects_effective_run(cli_run):
    """II.5: a consolidated 'Paths:' line records which paths actually ran. The bench args
    request backend 1 (CPU), so it must report backend=CPU, and carry the q15/cache/hybrid/
    proc fields. Fail-on-bug: the record drops or mis-reports the effective backend."""
    m = re.search(r"^Paths:\s*backend=(\S+)\s+q15=(\S+)\s+cache=(\S+)\s+hybrid=(\S+)\s+proc=",
                  cli_run, re.MULTILINE)
    assert m, "the consolidated Paths line is missing or malformed (plan §II.5)"
    assert m.group(1) == "CPU", f"effective backend should be CPU (requested), got {m.group(1)}"
    assert m.group(2) in ("on", "off", "fell-back-to-float")
    assert m.group(3) in ("hit", "built", "none")


def test_cache_mode_prints_paths_with_cache_state(desktop_binary, tmp_path):
    """II.5 code-review fix: in cache mode the consolidated Paths line is printed too (it was
    suppressed before — the cache-mode exit bypassed the normal render), and cache= reports
    the only state where it is meaningful (built/hit). Run in an isolated cwd so the
    output/cache dir is fresh -> first run builds. Fail-on-bug: drop the cache-mode Paths
    render and no `cache=built` line appears."""
    args = [str(desktop_binary), str(WAV), "--cache", *BENCH_ARGS]
    r = subprocess.run(args, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, f"cache run failed: {r.stderr[-1500:]}"
    out = r.stdout + r.stderr
    m = re.search(r"^Paths:.*cache=(\S+)", out, re.MULTILINE)
    assert m, "cache-mode run must still print the consolidated Paths line"
    assert m.group(1) in ("built", "hit"), f"cache state should be built/hit, got {m.group(1)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
