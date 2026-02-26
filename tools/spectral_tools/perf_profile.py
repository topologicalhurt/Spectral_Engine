"""Linux perf profiling helpers for benchmark stage analysis.

Design:
- `matrix` strategy works with current CLI by running stage-isolated commands:
  analysis-only via export backend, synthesis-heavy via cache-hit rendering.
- `markers` strategy validates native stage begin/end hooks emitted by the
  engine and reports marker contract coverage.
"""

from __future__ import annotations

import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark_types import TestStatus
from .constants import DEFAULT_PEEK_TAIL_LINES, STAGE_MARKER_LINE_RE
from .process import run
from .utils import tail_lines


DEFAULT_PERF_EVENTS = "cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock"
PERF_PARANOID_RE = re.compile(r"perf_event_paranoid setting is\s+(-?\d+)")
STAGE_TIMING_RE = re.compile(r"FFT:\s*([0-9]+(?:\.[0-9]+)?)ms\s*Track:\s*([0-9]+(?:\.[0-9]+)?)ms")
RENDER_TIMING_RE = re.compile(
    r"Render:\s*Synth\s*([0-9]+(?:\.[0-9]+)?)ms\s*Norm\s*([0-9]+(?:\.[0-9]+)?)ms\s*Write\s*([0-9]+(?:\.[0-9]+)?)ms"
)
MARKER_LINE_RE = STAGE_MARKER_LINE_RE  # module-local alias for readability
MARKER_STRICT_STAGES = ("fft", "track", "synth", "norm", "write")
MARKER_FALLBACK_STAGES = ("analysis", "synth", "norm", "write")
DEFAULT_PERF_CLI_ARGS = ["0", "1", "0", "4096", "128", "-85", "20", "0"]


@dataclass(slots=True)
class PerfCounter:
    event: str
    value: float | None
    unit: str | None
    raw_value: str


def _ns_to_ms(value_ns: int) -> float:
    return float(value_ns) / 1_000_000.0


def _duration_stats_ms(durations_ns: list[int]) -> dict[str, float | int]:
    if not durations_ns:
        return {"count": 0}
    ms = [_ns_to_ms(value) for value in durations_ns]
    return {
        "count": len(ms),
        "min_ms": min(ms),
        "median_ms": float(statistics.median(ms)),
        "mean_ms": float(statistics.mean(ms)),
        "max_ms": max(ms),
    }


def _parse_stage_markers(log_text: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(log_text.splitlines(), start=1):
        line = raw_line.strip()
        match = MARKER_LINE_RE.match(line)
        if not match:
            continue
        kind, stage, mono_ns_raw = match.groups()
        events.append(
            {
                "kind": kind.lower(),
                "stage": stage,
                "mono_ns": int(mono_ns_raw),
                "line": lineno,
                "raw": line,
            }
        )

    open_events: dict[str, list[dict[str, Any]]] = {}
    stage_spans_ns: dict[str, list[int]] = {}
    unmatched_ends: list[dict[str, Any]] = []

    for event in events:
        stage = str(event["stage"])
        if event["kind"] == "begin":
            open_events.setdefault(stage, []).append(event)
            continue

        stage_stack = open_events.get(stage)
        if stage_stack:
            begin_event = stage_stack.pop()
            duration_ns = int(event["mono_ns"]) - int(begin_event["mono_ns"])
            if duration_ns < 0:
                duration_ns = 0
            stage_spans_ns.setdefault(stage, []).append(duration_ns)
        else:
            unmatched_ends.append(event)

    unmatched_begins: list[dict[str, Any]] = []
    for stage_stack in open_events.values():
        unmatched_begins.extend(stage_stack)

    observed_stages = sorted(stage_spans_ns.keys())
    strict_missing = [stage for stage in MARKER_STRICT_STAGES if stage not in stage_spans_ns]
    effective_missing, analysis_fallback_used = _effective_missing_stages(
        strict_missing=strict_missing,
        observed_stage_spans=stage_spans_ns,
    )

    span_stats = {
        stage: _duration_stats_ms(durations)
        for stage, durations in stage_spans_ns.items()
        if durations
    }

    return {
        "events": events,
        "event_count": len(events),
        "stage_spans_ns": stage_spans_ns,
        "observed_stages": observed_stages,
        "stage_duration_stats_ms": span_stats,
        "strict_missing_stages": strict_missing,
        "missing_stages": effective_missing,
        "analysis_fallback_used": analysis_fallback_used,
        "unmatched_begins": unmatched_begins,
        "unmatched_ends": unmatched_ends,
    }


def _effective_missing_stages(
    *,
    strict_missing: list[str],
    observed_stage_spans: dict[str, list[int]],
) -> tuple[list[str], bool]:
    analysis_fallback_used = False
    effective_missing = list(strict_missing)
    if (
        "analysis" in observed_stage_spans
        and "fft" in effective_missing
        and "track" in effective_missing
    ):
        effective_missing = [stage for stage in effective_missing if stage not in {"fft", "track"}]
        analysis_fallback_used = True
    return effective_missing, analysis_fallback_used


def _parse_perf_value(raw: str) -> float | None:
    token = raw.strip()
    if not token or token.startswith("<"):
        return None
    token = token.replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def _parse_perf_stat_csv(stderr_text: str) -> dict[str, PerfCounter]:
    counters: dict[str, PerfCounter] = {}
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line or ";" not in line:
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 3:
            continue

        raw_value = parts[0]
        unit = parts[1] or None
        event = parts[2]
        if not event:
            continue

        counters[event] = PerfCounter(
            event=event,
            value=_parse_perf_value(raw_value),
            unit=unit,
            raw_value=raw_value,
        )
    return counters


def _read_perf_event_paranoid() -> int | None:
    path = Path("/proc/sys/kernel/perf_event_paranoid")
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _detect_perf_paranoid_from_stderr(stderr_text: str) -> int | None:
    match = PERF_PARANOID_RE.search(stderr_text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _perf_access_blocked(stderr_text: str) -> bool:
    token = "Access to performance monitoring and observability operations is limited"
    return token in stderr_text


def _run_perf_stat(
    *,
    cmd: list[str],
    cwd: Path,
    events: str,
    timeout_sec: int | None,
) -> dict[str, Any]:
    paranoid_value = _read_perf_event_paranoid()
    perf_path = shutil.which("perf")
    if perf_path is None:
        return {
            "status": TestStatus.UNAVAILABLE.value,
            "reason": "perf not found in PATH",
            "command": None,
            "stdout": "",
            "stderr": "",
            "counters": {},
            "returncode": 127,
            "perf_event_paranoid": paranoid_value,
        }

    perf_cmd = [
        perf_path,
        "stat",
        "-x",
        ";",
        "--no-big-num",
        "-e",
        events,
        "--",
        *cmd,
    ]

    result = run(perf_cmd, cwd=cwd, timeout_sec=timeout_sec, check=False)
    counters = _parse_perf_stat_csv(result.stderr)

    status = TestStatus.OK.value if result.returncode == 0 else TestStatus.FAILED.value
    reason: str | None = None
    if result.returncode != 0:
        detected = _detect_perf_paranoid_from_stderr(result.stderr)
        if detected is not None:
            paranoid_value = detected
        if _perf_access_blocked(result.stderr):
            status = TestStatus.BLOCKED.value
            reason = (
                f"perf access denied (perf_event_paranoid={paranoid_value})"
                if paranoid_value is not None
                else "perf access denied (missing CAP_PERFMON/CAP_SYS_PTRACE/CAP_SYS_ADMIN)"
            )
        else:
            reason = "perf stat failed"

    return {
        "status": status,
        "reason": reason,
        "command": perf_cmd,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "counters": {
            name: {
                "value": counter.value,
                "unit": counter.unit,
                "raw_value": counter.raw_value,
            }
            for name, counter in counters.items()
        },
        "returncode": result.returncode,
        "elapsed_sec": result.elapsed_sec,
        "perf_event_paranoid": paranoid_value,
    }


def _with_backend(cli_args: list[str], backend_id: int) -> list[str]:
    """Override the backend-id argument (positional index 7) in cli_args.

    Positional contract (DEFAULT_PERF_CLI_ARGS):
      0=verbose  1=format  2=threads  3=fft_size  4=hop_size
      5=db_thresh  6=min_dur_ms  7=backend_id
    """
    args = list(cli_args)
    defaults = DEFAULT_PERF_CLI_ARGS
    if len(args) < len(defaults):
        args = args + defaults[len(args) :]
    args[7] = str(backend_id)
    return args


def _parse_stage_output(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    fft_track_match = STAGE_TIMING_RE.search(stdout)
    if fft_track_match:
        metrics["fft_ms"] = float(fft_track_match.group(1))
        metrics["track_ms"] = float(fft_track_match.group(2))

    render_match = RENDER_TIMING_RE.search(stdout)
    if render_match:
        metrics["synth_ms"] = float(render_match.group(1))
        metrics["norm_ms"] = float(render_match.group(2))
        metrics["write_ms"] = float(render_match.group(3))

    return metrics


def profile_stage_matrix(
    *,
    binary: Path,
    input_wav: Path,
    cli_args: list[str],
    cwd: Path,
    events: str,
    timeout_sec: int | None = None,
    dry_run: bool = False,
    include_full: bool = False,
) -> dict[str, Any]:
    analysis_cmd = [str(binary), str(input_wav), *_with_backend(cli_args, 4)]
    synth_cmd = [str(binary), str(input_wav), *cli_args, "--cache"]
    full_cmd = [str(binary), str(input_wav), *cli_args]

    plan = {
        "analysis": analysis_cmd,
        "synthesis_cache_warmup": synth_cmd,
        "synthesis_profiled": synth_cmd,
    }
    if include_full:
        plan["full_profiled"] = full_cmd

    if dry_run:
        return {
            "strategy": "matrix",
            "status": TestStatus.SKIPPED.value,
            "plan": plan,
            "events": events,
        }

    stages: dict[str, Any] = {}

    analysis_result = _run_perf_stat(cmd=analysis_cmd, cwd=cwd, events=events, timeout_sec=timeout_sec)
    analysis_stage = {
        "status": analysis_result["status"],
        "perf": analysis_result,
        "runtime_metrics": _parse_stage_output(analysis_result["stdout"]),
    }
    stages["analysis"] = analysis_stage

    if analysis_result["status"] in {TestStatus.UNAVAILABLE.value, TestStatus.BLOCKED.value}:
        return {
            "strategy": "matrix",
            "status": analysis_result["status"],
            "events": events,
            "stages": stages,
            "plan": plan,
            "notes": [
                "Perf collection blocked before stage runs could be profiled.",
                "Requires lower perf_event_paranoid and/or CAP_PERFMON capability.",
            ],
        }

    warmup = run(synth_cmd, cwd=cwd, check=False)
    stages["synthesis_warmup"] = {
        "status": TestStatus.OK.value if warmup.returncode == 0 else TestStatus.FAILED.value,
        "command": synth_cmd,
        "returncode": warmup.returncode,
        "elapsed_sec": warmup.elapsed_sec,
    }

    synthesis_result = _run_perf_stat(cmd=synth_cmd, cwd=cwd, events=events, timeout_sec=timeout_sec)
    synthesis_stage = {
        "status": synthesis_result["status"],
        "perf": synthesis_result,
        "runtime_metrics": _parse_stage_output(synthesis_result["stdout"]),
    }
    stages["synthesis"] = synthesis_stage

    if include_full:
        full_result = _run_perf_stat(cmd=full_cmd, cwd=cwd, events=events, timeout_sec=timeout_sec)
        stages["full"] = {
            "status": full_result["status"],
            "perf": full_result,
            "runtime_metrics": _parse_stage_output(full_result["stdout"]),
        }

    any_failed = any(stage.get("status") == TestStatus.FAILED.value for stage in stages.values())
    any_blocked = any(stage.get("status") == TestStatus.BLOCKED.value for stage in stages.values())
    return {
        "strategy": "matrix",
        "status": (
            TestStatus.FAILED.value
            if any_failed
            else (TestStatus.BLOCKED.value if any_blocked else TestStatus.OK.value)
        ),
        "events": events,
        "stages": stages,
        "plan": plan,
        "notes": [
            "Matrix strategy isolates analysis via export backend and synthesis via cache-hit path.",
            "Per-stage FFT vs Track hardware counters need marker-based native stage hooks for precise gating.",
        ],
    }


def profile_stage_markers_contract(
    *,
    binary: Path,
    input_wav: Path,
    cli_args: list[str],
    cwd: Path,
    timeout_sec: int | None = None,
    dry_run: bool = False,
    probe_runs: int = 1,
    split_analysis_markers: bool = False,
) -> dict[str, Any]:
    cmd = [str(binary), str(input_wav), *cli_args]
    requested_probe_runs = max(1, int(probe_runs))
    marker_env = {"SPECTRAL_STAGE_SPLIT_ANALYSIS": "1"} if split_analysis_markers else {}
    required_contract = {
        "begin_line": "SPECTRAL_STAGE_BEGIN <stage_name> <mono_ns>",
        "end_line": "SPECTRAL_STAGE_END <stage_name> <mono_ns>",
        "strict_stages": list(MARKER_STRICT_STAGES),
        "accepted_stage_sets": [
            list(MARKER_STRICT_STAGES),
            list(MARKER_FALLBACK_STAGES),
        ],
    }

    if dry_run:
        return {
            "strategy": "markers",
            "status": TestStatus.SKIPPED.value,
            "reason": "dry-run",
            "required_contract": required_contract,
            "plan": {
                "marker_probe": cmd,
                "probe_runs": requested_probe_runs,
                "env_overrides": marker_env,
            },
        }

    aggregate_stage_spans_ns: dict[str, list[int]] = {}
    aggregate_observed_stages: set[str] = set()
    aggregate_event_count = 0
    aggregate_unmatched_begins = 0
    aggregate_unmatched_ends = 0
    runs_missing_stages: list[dict[str, Any]] = []
    runs_strict_missing_stages: list[dict[str, Any]] = []
    analysis_fallback_runs = 0
    failed_runs: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    probe_elapsed_sec_values: list[float] = []
    last_stdout = ""
    last_stderr = ""

    for run_index in range(1, requested_probe_runs + 1):
        result = run(
            cmd,
            cwd=cwd,
            env=marker_env if marker_env else None,
            timeout_sec=timeout_sec,
            check=False,
        )
        marker_parse = _parse_stage_markers(f"{result.stdout}\n{result.stderr}")
        last_stdout = result.stdout
        last_stderr = result.stderr
        probe_elapsed_sec_values.append(float(result.elapsed_sec))

        aggregate_event_count += int(marker_parse["event_count"])
        aggregate_observed_stages.update(marker_parse["observed_stages"])
        aggregate_unmatched_begins += len(marker_parse["unmatched_begins"])
        aggregate_unmatched_ends += len(marker_parse["unmatched_ends"])
        if marker_parse["analysis_fallback_used"]:
            analysis_fallback_runs += 1

        for stage, spans in marker_parse["stage_spans_ns"].items():
            aggregate_stage_spans_ns.setdefault(stage, []).extend(int(span) for span in spans)

        if marker_parse["missing_stages"]:
            runs_missing_stages.append(
                {
                    "run": run_index,
                    "missing_stages": marker_parse["missing_stages"],
                }
            )
        if marker_parse["strict_missing_stages"]:
            runs_strict_missing_stages.append(
                {
                    "run": run_index,
                    "strict_missing_stages": marker_parse["strict_missing_stages"],
                }
            )

        run_summary = {
            "run": run_index,
            "returncode": result.returncode,
            "elapsed_sec": result.elapsed_sec,
            "event_count": marker_parse["event_count"],
            "observed_stages": marker_parse["observed_stages"],
            "missing_stages": marker_parse["missing_stages"],
            "strict_missing_stages": marker_parse["strict_missing_stages"],
            "analysis_fallback_used": marker_parse["analysis_fallback_used"],
            "unmatched_begin_count": len(marker_parse["unmatched_begins"]),
            "unmatched_end_count": len(marker_parse["unmatched_ends"]),
        }
        run_summaries.append(run_summary)

        if result.returncode != 0:
            failed_runs.append(
                {
                    "run": run_index,
                    "returncode": result.returncode,
                    "elapsed_sec": result.elapsed_sec,
                    "stderr_tail": tail_lines(result.stderr, max_lines=DEFAULT_PEEK_TAIL_LINES),
                    "stdout_tail": tail_lines(result.stdout, max_lines=DEFAULT_PEEK_TAIL_LINES),
                }
            )

    aggregate_stage_duration_stats_ms = {
        stage: _duration_stats_ms(spans)
        for stage, spans in aggregate_stage_spans_ns.items()
        if spans
    }
    aggregate_strict_missing = [
        stage for stage in MARKER_STRICT_STAGES if stage not in aggregate_stage_spans_ns
    ]
    aggregate_missing, aggregate_analysis_fallback_used = _effective_missing_stages(
        strict_missing=aggregate_strict_missing,
        observed_stage_spans=aggregate_stage_spans_ns,
    )

    status = TestStatus.OK.value
    reason: str | None = None
    if failed_runs:
        status = TestStatus.FAILED.value
        reason = f"marker probe command failed in {len(failed_runs)} run(s)"
    elif aggregate_event_count == 0:
        status = TestStatus.BLOCKED.value
        reason = (
            "No stage marker lines were observed. Ensure the runtime emits "
            "SPECTRAL_STAGE_BEGIN/SPECTRAL_STAGE_END lines to stderr or stdout."
        )
    elif aggregate_unmatched_begins > 0 or aggregate_unmatched_ends > 0:
        status = TestStatus.BLOCKED.value
        reason = "Stage marker stream contains unmatched begin/end pairs."
    elif runs_missing_stages:
        status = TestStatus.BLOCKED.value
        reason = f"Missing required stage markers in {len(runs_missing_stages)} run(s)"

    return {
        "strategy": "markers",
        "status": status,
        "reason": reason,
        "required_contract": required_contract,
        "command": {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": 0 if not failed_runs else int(failed_runs[0]["returncode"]),
            "elapsed_sec_total": float(sum(probe_elapsed_sec_values)),
            "elapsed_sec_median": float(statistics.median(probe_elapsed_sec_values)),
            "probe_runs": requested_probe_runs,
            "env_overrides": marker_env,
        },
        "marker_summary": {
            "probe_runs": requested_probe_runs,
            "event_count": aggregate_event_count,
            "observed_stages": sorted(aggregate_observed_stages),
            "stage_duration_stats_ms": aggregate_stage_duration_stats_ms,
            "missing_stages": aggregate_missing,
            "strict_missing_stages": aggregate_strict_missing,
            "analysis_fallback_used": aggregate_analysis_fallback_used,
            "analysis_fallback_runs": analysis_fallback_runs,
            "unmatched_begin_count": aggregate_unmatched_begins,
            "unmatched_end_count": aggregate_unmatched_ends,
            "runs_with_missing_stages": runs_missing_stages,
            "runs_with_strict_missing_stages": runs_strict_missing_stages,
        },
        "runs": run_summaries,
        "failed_runs": failed_runs,
        "stderr_tail": tail_lines(last_stderr, max_lines=DEFAULT_PEEK_TAIL_LINES),
        "stdout_tail": tail_lines(last_stdout, max_lines=DEFAULT_PEEK_TAIL_LINES),
    }
