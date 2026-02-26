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

from .benchmark_runner import BenchmarkRunner
from .benchmark_types import BenchMode, PerfStrategy, TestStatus
from .constants import (
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
from .perf_profile import (
    DEFAULT_PERF_EVENTS,
    profile_stage_markers_contract,
    profile_stage_matrix,
)
from .process import ProcessResult, run
from .report_output import render_report, write_report_json
from .utils import find_repo_root, is_executable, parse_int_csv, tail_lines

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


def parse_int_list(raw: str) -> list[int]:
    return parse_int_csv(raw)


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

    try:
        cli_args = resolve_cli_args(args.binary_cli_args, args.bench_args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
        for chunk in parse_int_list(args.chunk_values):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spectral benchmark workflows.")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Print commands logically without executing")

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
    bench.add_argument("-o", "--json-out", default=DEFAULT_JSON_OUTPUT, help="Write JSON report to this file")
    bench.add_argument("-r", "--raw", action="store_true", help="Print raw JSON report to stdout")
    bench.add_argument("--no-color", action="store_true", help="Disable ANSI colors for pretty output")
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
    suite.add_argument("-o", "--json-out", default=DEFAULT_JSON_OUTPUT, help="Write JSON report to this file")
    suite.add_argument("-r", "--raw", action="store_true", help="Print raw JSON report to stdout")
    suite.add_argument("--no-color", action="store_true", help="Disable ANSI colors for pretty output")

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

    parser.error(f"unsupported command: {args.command}")
    return 2
