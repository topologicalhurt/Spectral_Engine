"""Workflow orchestration and CLI wiring for benchmark_spectral."""

from __future__ import annotations

import argparse
import platform
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# When executed directly (python benchmark_workflow.py), __package__ is unset.
# Patch sys.path once so relative imports work in both modes.
if __package__ in {None, ""}:
    _tools_root = str(Path(__file__).resolve().parents[2])
    if _tools_root not in sys.path:
        sys.path.insert(0, _tools_root)
    __package__ = "spectral_tools.testing"  # noqa: A001

from ..core.console import Console
from .benchmark_runner import BenchmarkRunner
from ..performance.benchmark_types import BenchMode, PerfStrategy, TestStatus
from ..core.constants import (
    DEFAULT_BENCHMARK_BINARY_GLOB,
    DEFAULT_BENCH_MODE,
    DEFAULT_BENCH_RUNS,
    DEFAULT_BUILD_TARGET,
    DEFAULT_CHUNK_VALUES,
    DEFAULT_PERF_TIMEOUT_SEC,
    DEFAULT_SHORT_TAIL_LINES,
    DEFAULT_SUITE_BENCH_ARGS,
    DEFAULT_SUITE_INPUT,
    DEFAULT_SUITE_RUNS,
    DEFAULT_UTF8_ENCODING,
    REMAINDER_DELIMITER,
)
from ..performance.perf_profile import (
    DEFAULT_PERF_EVENTS,
    profile_stage_markers_contract,
    profile_stage_matrix,
)
from ..core.cli import add_dry_run_flag, add_output_options
from ..core.process import ProcessResult, run
from ..core.report_output import render_report, write_report_json
from ..core.utils import find_repo_root, is_executable, parse_int_csv, tail_lines

@dataclass(frozen=True, slots=True)
class PrefetchProfile:
    name: str
    candidate_batch: int
    scan_distance: int
    lookahead: int


PREFETCH_PROFILES = (
    PrefetchProfile("prefetch-baseline", 128, 48, 12),
    PrefetchProfile("prefetch-latency", 64, 32, 8),
    PrefetchProfile("prefetch-throughput", 256, 64, 16),
)
DEFAULT_CACHE_DIR: str | None = None
DEFAULT_BENCH_ARGS: str | None = None
DEFAULT_JSON_OUTPUT: str | None = None


class Performance:
    """Runs shell utilities to configure/build/benchmark performance cases."""

    def __init__(
        self,
        repo_root: Path,
        dry_run: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.dry_run = dry_run
        self.runner = BenchmarkRunner(repo_root=self.repo_root, dry_run=dry_run)

    def run_shell_utility(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ProcessResult:
        run_cwd = (cwd or self.repo_root).resolve()

        if self.dry_run:
            return ProcessResult(cmd=cmd, cwd=str(run_cwd), returncode=0, stdout="", stderr="", elapsed_sec=0.0)

        return run(cmd, cwd=run_cwd, env=env, timeout_sec=timeout_sec, check=False)

    def _must_run(self, *args: Any, **kwargs: Any) -> ProcessResult:
        res = self.run_shell_utility(*args, **kwargs)
        if res.returncode != 0:
            raise RuntimeError(
                f"command failed (rc={res.returncode}): {' '.join(res.cmd)}\n"
                f"stderr:\n{tail_lines(res.stderr, max_lines=DEFAULT_SHORT_TAIL_LINES)}\n"
                f"stdout:\n{tail_lines(res.stdout, max_lines=DEFAULT_SHORT_TAIL_LINES)}"
            )
        return res

    def run_benchmark(
        self,
        *,
        binary: Path,
        input_wav: Path,
        runs: int,
        mode: str,
        cli_args: list[str],
        cache_dir: Path | None,
        reset_cache: bool,
    ) -> dict[str, Any]:
        result = self.runner.run(
            binary=binary,
            input_wav=input_wav,
            runs=runs,
            mode=mode,
            cli_args=cli_args,
            cache_dir=cache_dir,
            reset_cache=reset_cache,
        )
        return {
            "command": result.command,
            "stdout": result.stdout,
            "metrics": result.metrics,
        }

    def configure_build(self, build_dir: Path, cmake_args: list[str]) -> ProcessResult:
        cmd = ["cmake", "-S", ".", "-B", str(build_dir)] + cmake_args
        return self._must_run(cmd, cwd=self.repo_root)

    def build_target(self, build_dir: Path, target: str = DEFAULT_BUILD_TARGET) -> ProcessResult:
        cmd = ["cmake", "--build", str(build_dir), "--target", target, "--parallel"]
        return self._must_run(cmd, cwd=self.repo_root)

    def resolve_desktop_binary(self, build_dir: Path) -> Path:
        bin_dir = (build_dir / "bin").resolve()
        candidates = sorted(bin_dir.glob(DEFAULT_BENCHMARK_BINARY_GLOB))
        for path in candidates:
            if is_executable(path):
                return path
        raise RuntimeError(f"unable to resolve desktop binary in {bin_dir}")

    def collect_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "repo_root": str(self.repo_root),
            "python": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "perf_bin": shutil.which("perf"),
        }

        perf_paranoid_file = Path("/proc/sys/kernel/perf_event_paranoid")
        if perf_paranoid_file.exists():
            try:
                context["perf_event_paranoid"] = int(perf_paranoid_file.read_text(encoding=DEFAULT_UTF8_ENCODING).strip())
            except (OSError, ValueError):
                context["perf_event_paranoid"] = None

        for label, cmd in (
            ("uname", ["uname", "-srmo"]),
            ("nproc", ["nproc"]),
            ("git_head", ["git", "rev-parse", "--short", "HEAD"]),
        ):
            try:
                proc = run(cmd, cwd=self.repo_root, check=False)
            except OSError:
                continue
            if proc.returncode == 0:
                context[label] = proc.stdout.strip()

        return context

    def run_case(
        self,
        *,
        case_name: str,
        build_dir: Path,
        cmake_args: list[str],
        benchmark_input: Path,
        benchmark_runs: int,
        benchmark_mode: str,
        benchmark_cli_args: list[str],
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "build_dir": str(build_dir),
            "cmake_args": cmake_args,
            "benchmark_runs": benchmark_runs,
            "benchmark_mode": benchmark_mode,
            "benchmark_cli_args": benchmark_cli_args,
        }

        if self.dry_run:
            details["planned"] = {
                "configure_cmd": ["cmake", "-S", ".", "-B", str(build_dir), *cmake_args],
                "build_cmd": ["cmake", "--build", str(build_dir), "--target", DEFAULT_BUILD_TARGET, "--parallel"],
                "benchmark_cmd": [
                    str(build_dir / "bin" / "spectral_*_desktop"),
                    str(benchmark_input),
                    *benchmark_cli_args,
                    *(["--cache"] if benchmark_mode == BenchMode.CACHE.value else []),
                ],
            }
            return {
                "name": case_name,
                "status": TestStatus.SKIPPED.value,
                "summary": "dry-run (commands planned)",
                "details": details,
            }

        try:
            self.configure_build(build_dir, cmake_args)
            self.build_target(build_dir)
            binary = self.resolve_desktop_binary(build_dir)
            details["binary"] = str(binary)

            bench = self.run_benchmark(
                binary=binary,
                input_wav=benchmark_input,
                runs=benchmark_runs,
                mode=benchmark_mode,
                cli_args=benchmark_cli_args,
                cache_dir=None,
                reset_cache=False,
            )
            details["metrics"] = bench["metrics"]
            details["benchmark_cmd"] = bench["command"]

            track_median = bench["metrics"].get("track_series", {}).get("median_ms")
            track_bw = bench["metrics"].get("track_series", {}).get("bw_median_gibps")
            warm_total = bench["metrics"].get("summary", {}).get("warm_median_ms")
            summary = (
                f"track_median={track_median:.3f}ms, track_bw={track_bw:.3f}GiB/s, "
                f"warm_total={warm_total:.3f}ms"
                if isinstance(track_median, float)
                and isinstance(track_bw, float)
                and isinstance(warm_total, float)
                else "benchmark completed"
            )

            return {
                "name": case_name,
                "status": TestStatus.OK.value,
                "summary": summary,
                "details": details,
            }
        except Exception as exc:
            details["error"] = str(exc)
            return {
                "name": case_name,
                "status": TestStatus.FAILED.value,
                "summary": "case failed",
                "details": details,
            }





def parse_bench_args(raw: str) -> list[str]:
    return shlex.split(raw)


def resolve_cli_args(remainder: list[str], bench_args: str | None) -> list[str]:
    extra = list(remainder)
    if extra and extra[0] == REMAINDER_DELIMITER:
        extra = extra[1:]

    if extra and bench_args is not None:
        raise ValueError("Do not pass both '-- <args>' and --bench-args")

    if extra:
        return extra

    if bench_args is not None:
        return parse_bench_args(bench_args)

    return []


def format_single_summary(metrics: dict[str, Any]) -> str:
    summary = metrics.get("summary", {})
    track = metrics.get("track_series", {})
    stage = metrics.get("stage_medians", {})

    total_median = summary.get("median_ms")
    total_warm = summary.get("warm_median_ms")
    track_median = track.get("median_ms")
    track_bw = track.get("bw_median_gibps")
    synth_kernel = stage.get("synth_kernel_ms")
    if synth_kernel is None:
        synth_kernel = stage.get("synth_ms")
    synth_wall = stage.get("synth_wall_ms")

    if not isinstance(total_median, float) or not isinstance(track_median, float):
        return "benchmark completed"

    fields = [f"total_median={total_median:.3f}ms", f"track_median={track_median:.3f}ms"]
    if isinstance(total_warm, float):
        fields.append(f"total_warm_median={total_warm:.3f}ms")
    if isinstance(synth_kernel, float):
        fields.append(f"synth_kernel={synth_kernel:.3f}ms")
    if isinstance(synth_wall, float):
        fields.append(f"synth_wall={synth_wall:.3f}ms")
    if isinstance(track_bw, float):
        fields.append(f"track_bw={track_bw:.3f}GiB/s")
    return ", ".join(fields)


def collect_perf_profile(
    *,
    args: argparse.Namespace,
    perf: Performance,
    binary: Path,
    input_wav: Path,
    cli_args: list[str],
) -> dict[str, Any] | None:
    if not bool(args.perf):
        return None

    timeout_sec: int | None = None
    if int(args.perf_timeout_sec) > 0:
        timeout_sec = int(args.perf_timeout_sec)

    strategy = str(args.perf_strategy)
    if strategy == PerfStrategy.MARKERS.value:
        return profile_stage_markers_contract(
            binary=binary.resolve(),
            input_wav=input_wav.resolve(),
            cli_args=cli_args,
            cwd=perf.repo_root,
            timeout_sec=timeout_sec,
            dry_run=perf.dry_run,
            probe_runs=max(1, int(args.runs)),
            split_analysis_markers=True,
        )

    return profile_stage_matrix(
        binary=binary.resolve(),
        input_wav=input_wav.resolve(),
        cli_args=cli_args,
        cwd=perf.repo_root,
        events=str(args.perf_events),
        timeout_sec=timeout_sec,
        dry_run=perf.dry_run,
        include_full=bool(args.perf_include_full),
    )


def emit_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.json_out:
        write_report_json(Path(args.json_out), report)
    print(render_report(report, raw=bool(args.raw), use_color=not bool(args.no_color)))


def run_single_bench(args: argparse.Namespace, perf: Performance) -> int:
    binary = Path(args.binary)
    input_wav = Path(args.input)
    console = Console(use_color=not bool(args.no_color))

    try:
        cli_args = resolve_cli_args(args.binary_cli_args, args.bench_args)
    except ValueError as exc:
        console.print_error(str(exc))
        return 2

    try:
        result = perf.run_benchmark(
            binary=binary,
            input_wav=input_wav,
            runs=args.runs,
            mode=args.mode,
            cli_args=cli_args,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            reset_cache=args.reset_cache,
        )
    except Exception as exc:
        report = {
            "suite": "single-benchmark",
            "context": perf.collect_context(),
            "tests": [
                {
                    "name": "single",
                    "status": TestStatus.FAILED.value,
                    "summary": "benchmark failed",
                    "details": {
                        "binary": str(binary.resolve()),
                        "input": str(input_wav.resolve()),
                        "runs": args.runs,
                        "mode": args.mode,
                        "bench_args": cli_args,
                        "error": str(exc),
                    },
                }
            ],
        }
        emit_report(args, report)
        return 1

    perf_profile: dict[str, Any] | None = None
    try:
        perf_profile = collect_perf_profile(
            args=args,
            perf=perf,
            binary=binary,
            input_wav=input_wav,
            cli_args=cli_args,
        )
    except Exception as exc:
        perf_profile = {
            "strategy": str(args.perf_strategy),
            "status": TestStatus.FAILED.value,
            "reason": f"perf profile collection failed: {exc}",
        }

    test_status = TestStatus.OK.value
    perf_status_norm = str((perf_profile or {}).get("status", "")).strip().lower()
    if perf_status_norm == TestStatus.FAILED.value.lower():
        test_status = TestStatus.FAILED.value

    report = {
        "suite": "single-benchmark",
        "context": perf.collect_context(),
        "tests": [
            {
                "name": "single",
                "status": test_status,
                "summary": format_single_summary(result["metrics"]),
                "details": {
                    "binary": str(binary.resolve()),
                    "input": str(input_wav.resolve()),
                    "runs": args.runs,
                    "mode": args.mode,
                    "bench_args": cli_args,
                    "metrics": result["metrics"],
                    "command": result["command"],
                    "perf_profile": perf_profile,
                },
            }
        ],
    }
    emit_report(args, report)

    return 1 if test_status == TestStatus.FAILED.value else 0


def run_suite(args: argparse.Namespace, perf: Performance) -> int:
    benchmark_input = Path(args.input).resolve()
    bench_args = parse_bench_args(args.bench_args)

    tests: list[dict[str, Any]] = []

    if not args.skip_chunk_sweep:
        for chunk in parse_int_csv(args.chunk_values):
            case = perf.run_case(
                case_name=f"chunk={chunk}",
                build_dir=perf.repo_root / f"build_perf_chunk_{chunk}",
                cmake_args=[
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DSPECTRAL_DEBUG_LOGGING=OFF",
                    f"-DSPECTRAL_TRACK_PAIR_OMP_CHUNK={chunk}",
                ],
                benchmark_input=benchmark_input,
                benchmark_runs=args.runs,
                benchmark_mode=args.mode,
                benchmark_cli_args=bench_args,
            )
            tests.append(case)

    if not args.skip_prefetch_sweep:
        for profile in PREFETCH_PROFILES:
            cflags = (
                f"-DSPECTRAL_TRACK_CANDIDATE_BATCH={profile.candidate_batch} "
                f"-DSPECTRAL_TRACK_SCAN_PREFETCH_DISTANCE={profile.scan_distance} "
                f"-DSPECTRAL_TRACK_PREFETCH_LOOKAHEAD={profile.lookahead}"
            )
            case = perf.run_case(
                case_name=profile.name,
                build_dir=perf.repo_root / f"build_perf_{profile.name}",
                cmake_args=[
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DSPECTRAL_DEBUG_LOGGING=OFF",
                    "-DSPECTRAL_TRACK_PAIR_OMP_CHUNK=0",
                    f"-DCMAKE_C_FLAGS={cflags}",
                ],
                benchmark_input=benchmark_input,
                benchmark_runs=args.runs,
                benchmark_mode=args.mode,
                benchmark_cli_args=bench_args,
            )
            tests.append(case)

    report = {
        "suite": "spectral-performance-suite",
        "context": {
            **perf.collect_context(),
            "input": str(benchmark_input),
            "runs": args.runs,
            "mode": args.mode,
            "bench_args": bench_args,
        },
        "tests": tests,
    }

    emit_report(args, report)
    return 1 if any(t.get("status") == TestStatus.FAILED.value for t in tests) else 0


def run_m7_census(args: argparse.Namespace, perf: Performance) -> int:
    """Embedded Layer 0/2: codegen census [measured] + mca loop cycles [modeled]."""
    from ..performance.embedded import codegen, toolchain

    out_dir = Path(args.out_dir) if args.out_dir else perf.repo_root / "build" / "perf_model"
    test: dict[str, Any]
    try:
        tc = toolchain.discover(perf.repo_root, need=frozenset({"mca"}))
        result = codegen.codegen_report(tc, out_dir=out_dir)
        status = TestStatus.OK.value if not result.failed_regions else TestStatus.BLOCKED.value
        loops_summary = ", ".join(
            f"{loop.kernel}/{loop.label}={loop.cycles_per_iter:.1f}cyc" for loop in result.loops
        )
        test = {
            "name": "m7-census",
            "status": status,
            "summary": loops_summary or "census only",
            "details": result.as_dict(),
        }
    except (toolchain.ToolchainError, codegen.CodegenError) as exc:
        test = {
            "name": "m7-census",
            "status": TestStatus.FAILED.value,
            "summary": "census failed",
            "details": {"error": str(exc)},
        }

    report = {"suite": "m7-perf-model", "context": perf.collect_context(), "tests": [test]}
    emit_report(args, report)
    return 1 if test["status"] == TestStatus.FAILED.value else 0


def run_m7_mca_validate(args: argparse.Namespace, perf: Performance) -> int:
    """Layer 2 validation: mca CortexM7Model vs hand-derived expectations."""
    from ..performance.embedded import codegen, mca_validation, toolchain

    out_dir = Path(args.out_dir) if args.out_dir else perf.repo_root / "build" / "perf_model"
    test: dict[str, Any]
    try:
        tc = toolchain.discover(perf.repo_root, need=frozenset({"mca"}))
        report = mca_validation.validate(tc, out_dir=out_dir)
        summary = (
            f"max body delta {report.max_abs_body_delta * 100:.1f}% over "
            f"{len(report.results)} cases; back-edge divergence "
            f"+{report.mean_backedge_divergence:.1f} cyc/iter"
        )
        test = {
            "name": "m7-mca-validate",
            "status": TestStatus.OK.value,
            "summary": summary,
            "details": report.as_dict(),
        }
    except (toolchain.ToolchainError, codegen.CodegenError) as exc:
        test = {
            "name": "m7-mca-validate",
            "status": TestStatus.FAILED.value,
            "summary": "validation failed",
            "details": {"error": str(exc)},
        }

    report_doc = {"suite": "m7-perf-model", "context": perf.collect_context(), "tests": [test]}
    emit_report(args, report_doc)
    return 1 if test["status"] == TestStatus.FAILED.value else 0


def run_m7_counts(args: argparse.Namespace, perf: Performance) -> int:
    """Embedded Layer 1: dynamic instruction/byte counts under QEMU [measured]."""
    from ..performance.embedded import counts, toolchain

    out_dir = Path(args.out_dir) if args.out_dir else perf.repo_root / "build" / "perf_model"
    test: dict[str, Any]
    try:
        tc = toolchain.discover(perf.repo_root, need=frozenset({"qemu"}))
        result = counts.measure(
            tc, out_dir=out_dir, verify_reproducible=not args.no_verify
        )
        process_counts = result.range("spectral_arm32_process")
        summary = (
            f"process: {process_counts.insns} insns, "
            f"{process_counts.load_bytes}B ld / {process_counts.store_bytes}B st"
            if process_counts
            else "counts collected"
        )
        test = {
            "name": "m7-counts",
            "status": TestStatus.OK.value,
            "summary": summary,
            "details": result.as_dict(),
        }
    except (toolchain.ToolchainError, counts.CountsError) as exc:
        test = {
            "name": "m7-counts",
            "status": TestStatus.FAILED.value,
            "summary": "counts failed",
            "details": {"error": str(exc)},
        }

    report = {"suite": "m7-perf-model", "context": perf.collect_context(), "tests": [test]}
    emit_report(args, report)
    return 1 if test["status"] == TestStatus.FAILED.value else 0


def run_m7_bootstrap(args: argparse.Namespace, perf: Performance) -> int:
    """Download + sha-verify + extract the pinned newlib toolchain."""
    from ..performance.embedded import toolchain

    console = Console(use_color=not bool(args.no_color))
    try:
        path = toolchain.bootstrap(perf.repo_root, force=bool(args.force))
    except toolchain.ToolchainError as exc:
        console.print_error(str(exc))
        return 1
    print(f"toolchain ready: {path}")
    return 0


def run_m7_stalls(args: argparse.Namespace, perf: Performance) -> int:
    """Layer 3 (P4): memory-stall bounds modeled over the measured trace."""
    from ..performance.embedded import counts, memory_model, toolchain

    out_dir = Path(args.out_dir) if args.out_dir else perf.repo_root / "build" / "perf_model"
    test: dict[str, Any]
    try:
        tc = toolchain.discover(perf.repo_root, need=frozenset({"qemu"}))
        trace_path = out_dir / "qemu_trace.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        # The trace run is slower; counts determinism is m7-counts' contract.
        counts.measure(tc, out_dir=out_dir, verify_reproducible=False,
                       trace_path=trace_path)
        report = memory_model.analyze_trace(trace_path)
        doc = report.as_dict()
        mean = doc["per_block_mean_stalls"]
        summary = (
            f"{doc['n_blocks']} epochs; mean stalls/block "
            f"[{mean['bandwidth_bound']}, {mean['serial_bound']}] cyc; "
            f"fills {report.total.fills_row_hit}+{report.total.fills_row_miss}rm"
        )
        test = {
            "name": "m7-stalls",
            "status": TestStatus.OK.value,
            "summary": summary,
            "details": doc,
        }
    except (toolchain.ToolchainError, counts.CountsError, OSError) as exc:
        test = {
            "name": "m7-stalls",
            "status": TestStatus.FAILED.value,
            "summary": "memory model failed",
            "details": {"error": str(exc)},
        }

    report_doc = {"suite": "m7-perf-model", "context": perf.collect_context(), "tests": [test]}
    emit_report(args, report_doc)
    return 1 if test["status"] == TestStatus.FAILED.value else 0


def run_m7_wcet(args: argparse.Namespace, perf: Performance) -> int:
    """P5: per-block WCET upper bound composed from the validated stack."""
    from ..performance.embedded import codegen, counts, toolchain, wcet

    out_dir = Path(args.out_dir) if args.out_dir else perf.repo_root / "build" / "perf_model"
    test: dict[str, Any]
    try:
        tc = toolchain.discover(perf.repo_root, need=frozenset({"mca", "qemu"}))
        report = wcet.wcet(
            tc, out_dir=out_dir, active=args.active,
            scan_segments=args.scan_segments, block=args.block,
        )
        doc = report.as_dict()
        summary = (
            f"WCET {doc['wcet_block_cycles']:.0f} cyc/block "
            f"({args.active} active, {args.scan_segments} scanned) = "
            f"{doc['budget_fraction'] * 100:.1f}% of the 48k block budget"
        )
        test = {
            "name": "m7-wcet",
            "status": TestStatus.OK.value,
            "summary": summary,
            "details": doc,
        }
    except (toolchain.ToolchainError, codegen.CodegenError,
            counts.CountsError, wcet.WcetError) as exc:
        test = {
            "name": "m7-wcet",
            "status": TestStatus.FAILED.value,
            "summary": "wcet failed",
            "details": {"error": str(exc)},
        }

    report_doc = {"suite": "m7-perf-model", "context": perf.collect_context(), "tests": [test]}
    emit_report(args, report_doc)
    return 1 if test["status"] == TestStatus.FAILED.value else 0


def run_m7_baseline(args: argparse.Namespace, perf: Performance) -> int:
    """Performance baseline: --generate freezes; default verifies (the gate)."""
    from ..performance.embedded import codegen, counts, expectations, toolchain, wcet

    out_dir = Path(args.out_dir) if args.out_dir else perf.repo_root / "build" / "perf_model"
    test: dict[str, Any]
    try:
        tc = toolchain.discover(perf.repo_root, need=frozenset({"mca", "qemu"}))
        if args.generate:
            path = expectations.generate(tc, out_dir=out_dir)
            test = {
                "name": "m7-baseline",
                "status": TestStatus.OK.value,
                "summary": f"baseline re-signed: {path.relative_to(perf.repo_root)}",
                "details": {"path": str(path)},
            }
        else:
            fails = expectations.verify(tc, out_dir=out_dir)
            test = {
                "name": "m7-baseline",
                "status": TestStatus.OK.value if not fails else TestStatus.FAILED.value,
                "summary": ("gate PASS: live stack within tolerances + stone ceilings"
                            if not fails else f"gate FAIL: {len(fails)} quantities moved"),
                "details": {"failures": fails},
            }
    except (toolchain.ToolchainError, codegen.CodegenError, counts.CountsError,
            wcet.WcetError, expectations.ExpectationsError) as exc:
        test = {
            "name": "m7-baseline",
            "status": TestStatus.FAILED.value,
            "summary": "baseline failed",
            "details": {"error": str(exc)},
        }

    report_doc = {"suite": "m7-perf-model", "context": perf.collect_context(), "tests": [test]}
    emit_report(args, report_doc)
    return 1 if test["status"] == TestStatus.FAILED.value else 0


def run_measure(args: argparse.Namespace, perf: Performance) -> int:
    """Matrix-driven orchestration: probe or run the instruments for a target.

    `measure --list` shows the full target×instrument matrix with live
    availability probes. `measure --target <name>` runs every available
    instrument for that target (building first where the profile names a
    cmake target and --build is given) and emits one combined report.
    """
    from ..performance import matrix

    if args.list or args.target is None:
        probed = matrix.probe_matrix()
        report = {
            "suite": "measurement-matrix",
            "context": perf.collect_context(),
            "tests": [
                {
                    "name": name,
                    "status": TestStatus.OK.value,
                    "summary": entry["description"],
                    "details": {
                        "build_target": entry["build_target"],
                        "instruments": entry["instruments"],
                    },
                }
                for name, entry in probed.items()
            ],
        }
        emit_report(args, report)
        return 0

    try:
        profile = matrix.target(args.target)
    except KeyError as exc:
        Console(use_color=not bool(args.no_color)).print_error(str(exc))
        return 2

    if args.build and profile.build_target is not None:
        build_dir = perf.repo_root / "build"
        perf.configure_build(build_dir, ["-DCMAKE_BUILD_TYPE=Release"])
        perf.build_target(build_dir, profile.build_target)

    if profile.name == "m7":
        rc_census = run_m7_census(args, perf)
        rc_counts = run_m7_counts(args, perf)
        return rc_census or rc_counts

    if profile.name in {"desktop", "simulate"}:
        if not args.input:
            Console(use_color=not bool(args.no_color)).print_error(
                f"measure --target {profile.name} needs --input <wav> (and a built binary)"
            )
            return 2
        build_dir = perf.repo_root / "build"
        binary = perf.resolve_desktop_binary(build_dir)
        args.binary = str(binary)
        args.binary_cli_args = []
        return run_single_bench(args, perf)

    Console(use_color=not bool(args.no_color)).print_error(
        f"target '{profile.name}' has no runnable instruments yet (all planned)"
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spectral benchmark workflows.")
    add_dry_run_flag(parser, short="-d")

    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Run single benchmark")
    bench.add_argument("-b", "--binary", required=True, help="Path to benchmark binary")
    bench.add_argument("-i", "--input", required=True, help="Path to WAV input")
    bench.add_argument("-n", "--runs", type=int, default=DEFAULT_BENCH_RUNS, help="Benchmark run count")
    bench.add_argument(
        "-m",
        "--mode",
        default=DEFAULT_BENCH_MODE,
        choices=[mode.value for mode in BenchMode],
        help="Benchmark mode",
    )
    bench.add_argument("-c", "--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory for cache mode")
    bench.add_argument("--reset-cache", action="store_true", help="Reset cache files before cache benchmark")
    bench.add_argument("-a", "--bench-args", default=DEFAULT_BENCH_ARGS, help="Binary args as a shell-quoted string")
    bench.add_argument("-P", "--perf", action="store_true", help="Collect Linux perf statistics")
    bench.add_argument(
        "--perf-strategy",
        default=PerfStrategy.MARKERS.value,
        choices=[strategy.value for strategy in PerfStrategy],
        help="Perf profiling strategy (markers runs --runs marker probes and enables split fft/track markers)",
    )
    bench.add_argument(
        "--perf-events",
        default=DEFAULT_PERF_EVENTS,
        help="Comma-separated perf events list",
    )
    bench.add_argument(
        "--perf-include-full",
        action="store_true",
        help="Include an extra full-pipeline perf run in matrix mode",
    )
    bench.add_argument(
        "--perf-timeout-sec",
        type=int,
        default=DEFAULT_PERF_TIMEOUT_SEC,
        help="Timeout for each perf-instrumented run (0 disables timeout)",
    )
    add_output_options(bench)
    bench.add_argument(
        "binary_cli_args",
        nargs=argparse.REMAINDER,
        help="Binary args passed through after '--' (standard delimiter for positional passthrough).",
    )

    suite = sub.add_parser("suite", help="Run chunk/prefetch/block performance sweep suite")
    suite.add_argument(
        "-i",
        "--input",
        default=DEFAULT_SUITE_INPUT,
        help="Path to WAV input",
    )
    suite.add_argument("-n", "--runs", type=int, default=DEFAULT_SUITE_RUNS, help="Benchmark run count per case")
    suite.add_argument(
        "-m",
        "--mode",
        default=DEFAULT_BENCH_MODE,
        choices=[mode.value for mode in BenchMode],
        help="Benchmark mode",
    )
    suite.add_argument(
        "-a",
        "--bench-args",
        default=DEFAULT_SUITE_BENCH_ARGS,
        help="CLI args forwarded to benchmark binary",
    )
    suite.add_argument("--chunk-values", default=DEFAULT_CHUNK_VALUES, help="Comma list for chunk sweep")
    suite.add_argument("--skip-chunk-sweep", action="store_true", help="Skip chunk sweep")
    suite.add_argument("--skip-prefetch-sweep", action="store_true", help="Skip prefetch profile sweep")
    add_output_options(suite)

    m7_census = sub.add_parser(
        "m7-census",
        help="Embedded M7: codegen instruction census [measured] + llvm-mca loop cycles [modeled]",
    )
    m7_census.add_argument("--out-dir", default=None, help="Artifact dir (default build/perf_model)")
    add_output_options(m7_census)

    m7_validate = sub.add_parser(
        "m7-mca-validate",
        help="Embedded M7: validate the llvm-mca CortexM7Model against hand-derived "
             "TRM + community-measured timings (P3 microbench set)",
    )
    m7_validate.add_argument("--out-dir", default=None, help="Artifact dir (default build/perf_model)")
    add_output_options(m7_validate)

    m7_counts = sub.add_parser(
        "m7-counts",
        help="Embedded M7: dynamic instruction/byte counts under QEMU mps2-an500 [measured]",
    )
    m7_counts.add_argument("--out-dir", default=None, help="Artifact dir (default build/perf_model)")
    m7_counts.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the duplicate run that asserts bit-identical counts",
    )
    add_output_options(m7_counts)

    m7_baseline = sub.add_parser(
        "m7-baseline",
        help="Embedded M7: verify the live stack against the frozen performance "
             "baseline + set-in-stone capacity ceilings; --generate re-signs it",
    )
    m7_baseline.add_argument("--generate", action="store_true",
                             help="Freeze current live numbers as the baseline (deliberate act)")
    m7_baseline.add_argument("--out-dir", default=None, help="Artifact dir (default build/perf_model)")
    add_output_options(m7_baseline)

    m7_wcet = sub.add_parser(
        "m7-wcet",
        help="Embedded M7: parametric per-block WCET upper bound [modeled+measured] "
             "composed from the validated stack (P5)",
    )
    m7_wcet.add_argument("--active", type=int, default=64, help="Worst-case active voices")
    m7_wcet.add_argument("--scan-segments", type=int, default=1024,
                         help="Worst-case segments scanned per block")
    m7_wcet.add_argument("--block", type=int, default=256, help="Block size in samples")
    m7_wcet.add_argument("--out-dir", default=None, help="Artifact dir (default build/perf_model)")
    add_output_options(m7_wcet)

    m7_stalls = sub.add_parser(
        "m7-stalls",
        help="Embedded M7: memory-stall bounds [modeled] from the QEMU address trace "
             "(P4 layer; constants calibrated from libDaisy-as-shipped + TRM + AN4891)",
    )
    m7_stalls.add_argument("--out-dir", default=None, help="Artifact dir (default build/perf_model)")
    add_output_options(m7_stalls)

    measure = sub.add_parser(
        "measure",
        help="Matrix-driven measurement: --list shows target×instrument availability; "
             "--target runs every available instrument for one build target",
    )
    measure.add_argument("--list", action="store_true", help="Probe and show the measurement matrix")
    measure.add_argument("--target", default=None, help="Measurement target (see --list)")
    measure.add_argument("--build", action="store_true", help="Configure+build the target's cmake target first")
    measure.add_argument("-i", "--input", default=None, help="WAV input (desktop/simulate targets)")
    measure.add_argument("-n", "--runs", type=int, default=DEFAULT_BENCH_RUNS, help="Benchmark run count")
    measure.add_argument("-m", "--mode", default=DEFAULT_BENCH_MODE,
                         choices=[mode.value for mode in BenchMode], help="Benchmark mode")
    measure.add_argument("-a", "--bench-args", default=DEFAULT_BENCH_ARGS,
                         help="Binary args as a shell-quoted string")
    measure.add_argument("-c", "--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory for cache mode")
    measure.add_argument("--reset-cache", action="store_true", help="Reset cache files before cache benchmark")
    measure.add_argument("-P", "--perf", action="store_true", help="Collect Linux perf statistics (desktop)")
    measure.add_argument("--perf-strategy", default=PerfStrategy.MARKERS.value,
                         choices=[strategy.value for strategy in PerfStrategy], help="Perf profiling strategy")
    measure.add_argument("--perf-events", default=DEFAULT_PERF_EVENTS, help="Comma-separated perf events list")
    measure.add_argument("--perf-include-full", action="store_true",
                         help="Include an extra full-pipeline perf run in matrix mode")
    measure.add_argument("--perf-timeout-sec", type=int, default=DEFAULT_PERF_TIMEOUT_SEC,
                         help="Timeout for each perf-instrumented run (0 disables timeout)")
    measure.add_argument("--out-dir", default=None, help="Artifact dir for m7 instruments")
    measure.add_argument("--no-verify", action="store_true",
                         help="m7: skip the duplicate run that asserts bit-identical counts")
    add_output_options(measure)

    m7_bootstrap = sub.add_parser(
        "m7-bootstrap",
        help="Download + sha-verify the pinned newlib arm-none-eabi toolchain into tools/toolchains/",
    )
    m7_bootstrap.add_argument("--force", action="store_true", help="Re-download even if present")
    add_output_options(m7_bootstrap)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv or sys.argv[1:])

    repo_root = find_repo_root(Path(__file__))
    perf = Performance(repo_root=repo_root, dry_run=args.dry_run)

    if args.command == "bench":
        return run_single_bench(args, perf)
    if args.command == "suite":
        return run_suite(args, perf)
    if args.command == "m7-census":
        return run_m7_census(args, perf)
    if args.command == "m7-mca-validate":
        return run_m7_mca_validate(args, perf)
    if args.command == "m7-counts":
        return run_m7_counts(args, perf)
    if args.command == "m7-stalls":
        return run_m7_stalls(args, perf)
    if args.command == "m7-wcet":
        return run_m7_wcet(args, perf)
    if args.command == "m7-baseline":
        return run_m7_baseline(args, perf)
    if args.command == "measure":
        return run_measure(args, perf)
    if args.command == "m7-bootstrap":
        return run_m7_bootstrap(args, perf)

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
