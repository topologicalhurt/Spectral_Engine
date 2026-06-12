"""Measurement matrix: the SSOT for which instrument measures which build.

Every supported build target maps to a set of measurement instruments with
explicit availability probes. This is the answer to "use the correct tools
for different builds": one declarative table, probed at runtime, consumed by
the `measure` CLI verb — not scattered tool choices.

The pipeline shape every instrument follows (maintainer contract):
  build reproducibly (cmake, recorded args) -> run under a wrapper/observer
  -> parse -> interpret. In-process stage markers are the timing primitive on
  hosts (CLOCK_MONOTONIC ns, ~sub-µs cost); debuggers are deliberately NOT a
  timing instrument (a ptrace/breakpoint stop costs orders of magnitude more
  than a marker — see ADR-0002) but remain the right tool for state
  inspection. On the cross-M7 target, time does not exist on the host at
  all: QEMU TCG yields exact counts [measured], llvm-mca yields pipeline
  cycles [modeled], and the two are never blended.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class Availability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PLANNED = "planned"            # designed but not wired yet


@dataclass(frozen=True, slots=True)
class Instrument:
    name: str
    kind: str                      # timing | counters | memory | counts | model
    provenance: str                # measured | modeled
    probe: Callable[[], tuple[Availability, str]]
    cli: str                       # how to invoke it through benchmark_workflow


@dataclass(frozen=True, slots=True)
class TargetProfile:
    name: str
    description: str
    build_target: str | None       # cmake target, None = built by the rig itself
    instruments: tuple[Instrument, ...]


def _probe_always(reason: str = "") -> Callable[[], tuple[Availability, str]]:
    return lambda: (Availability.AVAILABLE, reason)


def _probe_planned(reason: str) -> Callable[[], tuple[Availability, str]]:
    return lambda: (Availability.PLANNED, reason)


def _probe_which(binary: str, hint: str) -> Callable[[], tuple[Availability, str]]:
    def probe() -> tuple[Availability, str]:
        if shutil.which(binary):
            return Availability.AVAILABLE, binary
        return Availability.UNAVAILABLE, hint
    return probe


def _probe_linux_perf() -> tuple[Availability, str]:
    if platform.system() != "Linux":
        return Availability.UNAVAILABLE, "Linux-only (use stage markers + time on macOS)"
    if shutil.which("perf"):
        return Availability.AVAILABLE, "perf"
    return Availability.UNAVAILABLE, "install linux-tools / perf"


def _probe_embedded(need: str) -> Callable[[], tuple[Availability, str]]:
    def probe() -> tuple[Availability, str]:
        from .embedded import toolchain

        try:
            toolchain.discover(need=frozenset({need} if need else set()))
        except toolchain.ToolchainError as exc:
            return Availability.UNAVAILABLE, str(exc)
        return Availability.AVAILABLE, need or "arm-none-eabi-gcc"
    return probe


STAGE_MARKERS = Instrument(
    name="stage-markers",
    kind="timing",
    provenance="measured",
    probe=_probe_always("in-process CLOCK_MONOTONIC ns markers (spectral_log.h)"),
    cli="bench -P --perf-strategy markers",
)
RSS_TIME = Instrument(
    name="rss-time",
    kind="memory",
    provenance="measured",
    probe=_probe_always("GNU time -f %M or BSD time -l (flavor probed)"),
    cli="bench (always collected)",
)
LINUX_PERF = Instrument(
    name="perf-stat",
    kind="counters",
    provenance="measured",
    probe=_probe_linux_perf,
    cli="bench -P --perf-strategy matrix",
)
XCTRACE = Instrument(
    name="xctrace",
    kind="counters",
    provenance="measured",
    probe=_probe_planned("macOS Instruments CLI profile; evaluate when counter data is needed on Darwin"),
    cli="(planned)",
)
M7_QEMU_COUNTS = Instrument(
    name="qemu-counts",
    kind="counts",
    provenance="measured",
    probe=_probe_embedded("qemu"),
    cli="m7-counts",
)
M7_MCA_CENSUS = Instrument(
    name="mca-census",
    kind="model",
    provenance="modeled",
    probe=_probe_embedded("mca"),
    cli="m7-census",
)
M7_MCA_VALIDATION = Instrument(
    name="mca-validation",
    kind="model",
    provenance="modeled",
    probe=_probe_embedded("mca"),
    cli="m7-mca-validate",
)
M7_MEMORY_STALLS = Instrument(
    name="memory-stalls",
    kind="memory",
    provenance="modeled",
    probe=_probe_embedded("qemu"),
    cli="m7-stalls",
)
M7_GDBSTUB_STAGES = Instrument(
    name="qemu-gdbstub-stages",
    kind="counts",
    provenance="measured",
    probe=_probe_planned(
        "stage segmentation via qemu -s breakpoints: guest time freezes while "
        "stopped, so segmentation is distortion-free; add when per-stage "
        "attribution beyond symbol ranges is needed"
    ),
    cli="(planned)",
)
DAISY_ON_TARGET = Instrument(
    name="daisy-dwt-cycles",
    kind="counters",
    provenance="measured",
    probe=_probe_planned("DWT->CYCCNT on hardware; no board present — the calibration anchor when one exists"),
    cli="(planned)",
)


TARGETS: tuple[TargetProfile, ...] = (
    TargetProfile(
        name="desktop",
        description="Host desktop build (Metal/vDSP on macOS, CUDA on Linux x86)",
        build_target="desktop",
        instruments=(STAGE_MARKERS, RSS_TIME, LINUX_PERF, XCTRACE),
    ),
    TargetProfile(
        name="simulate",
        description="Host-built embedded simulation (Q15, single-threaded)",
        build_target="simulate",
        instruments=(STAGE_MARKERS, RSS_TIME, LINUX_PERF),
    ),
    TargetProfile(
        name="m7",
        description="Cross Cortex-M7 model (no hardware): counts measured, cycles modeled",
        build_target=None,
        instruments=(M7_QEMU_COUNTS, M7_MCA_CENSUS, M7_MCA_VALIDATION,
                     M7_MEMORY_STALLS, M7_GDBSTUB_STAGES),
    ),
    TargetProfile(
        name="daisy",
        description="On-target Daisy Seed STM32H750 (future: hardware calibration anchor)",
        build_target="daisy",
        instruments=(DAISY_ON_TARGET,),
    ),
)


def target(name: str) -> TargetProfile:
    for profile in TARGETS:
        if profile.name == name:
            return profile
    raise KeyError(f"unknown measurement target '{name}' (have: {[t.name for t in TARGETS]})")


def probe_matrix() -> dict[str, Any]:
    """Probe every instrument of every target; the `measure --list` view."""
    report: dict[str, Any] = {}
    for profile in TARGETS:
        rows = []
        for instrument in profile.instruments:
            availability, detail = instrument.probe()
            rows.append(
                {
                    "instrument": instrument.name,
                    "kind": instrument.kind,
                    "provenance": instrument.provenance,
                    "availability": availability.value,
                    "detail": detail,
                    "cli": instrument.cli,
                }
            )
        report[profile.name] = {
            "description": profile.description,
            "build_target": profile.build_target,
            "instruments": rows,
        }
    return report
