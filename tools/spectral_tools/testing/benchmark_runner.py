"""Benchmark execution runner for single-binary performance sweeps."""

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from os import environ
from pathlib import Path

from .benchmark_parsing import BenchmarkOutputParser
from ..performance.benchmark_types import BenchMode, BenchRunResult
from ..core.constants import (
    DEFAULT_LONG_TAIL_LINES,
    DEFAULT_PROCFS_SAMPLE_SEC,
    DEFAULT_UTF8_ENCODING,
)
from ..core.utils import fmt_float, is_executable, tail_lines

PROC_STATUS_ROOT = Path("/proc")
PROCFS_SAMPLE_ENV = "SPECTRAL_BENCH_PROCFS_SAMPLE_SEC"
TIME_CANDIDATES = (Path("/usr/bin/time"), Path("/bin/time"))
BSD_TIME_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)
CACHE_FLAG = "--cache"
CACHE_FILES = ("seg_cache_index.bin", "seg_cache_data.bin")
METRIC_SECTION_KEYS = ("summary", "stage_medians", "bandwidth_medians", "memory", "track_series")


@dataclass(slots=True)
class _RunAccumulator:
    totals: list[float] = field(default_factory=list)
    ffts: list[float] = field(default_factory=list)
    tracks: list[float] = field(default_factory=list)
    synth_kernels: list[float] = field(default_factory=list)
    synth_walls: list[float] = field(default_factory=list)
    norms: list[float] = field(default_factory=list)
    segbin_totals: list[float] = field(default_factory=list)
    rss_kb: list[float] = field(default_factory=list)
    fft_bandwidths: list[float] = field(default_factory=list)
    track_bandwidths: list[float] = field(default_factory=list)
    fft_bandwidth_rel_pcts: list[float] = field(default_factory=list)
    track_bandwidth_rel_pcts: list[float] = field(default_factory=list)
    first_total: float | None = None


def _detect_time_flavor() -> tuple[str | None, str | None]:
    """Return (path, flavor) for the system time binary.

    flavor is "gnu" (supports ``-f %M``, RSS in KB) or "bsd" (macOS, supports
    ``-l``, RSS in bytes). Executability alone does not imply GNU: macOS ships
    a BSD time at /usr/bin/time that rejects ``-f`` — probe, don't assume.
    """
    for candidate in TIME_CANDIDATES:
        if not is_executable(candidate):
            continue
        try:
            probe = subprocess.run(
                [str(candidate), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except OSError:
            continue
        if "GNU" in (probe.stdout + probe.stderr):
            return str(candidate), "gnu"
        try:
            probe = subprocess.run(
                [str(candidate), "-l", "true"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except OSError:
            continue
        if probe.returncode == 0 and "maximum resident set size" in probe.stderr:
            return str(candidate), "bsd"
    return None, None


@dataclass(slots=True)
class BenchmarkRunner:
    repo_root: Path
    dry_run: bool = False
    parser: BenchmarkOutputParser = field(init=False)
    time_bin: str | None = field(init=False, default=None)
    time_flavor: str | None = field(init=False, default=None)
    have_procfs: bool = field(init=False, default=False)
    procfs_sample_sec: float = field(init=False, default=DEFAULT_PROCFS_SAMPLE_SEC)

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        self.parser = BenchmarkOutputParser()
        self.time_bin, self.time_flavor = _detect_time_flavor()
        self.have_procfs = PROC_STATUS_ROOT.is_dir()
        try:
            self.procfs_sample_sec = float(environ.get(PROCFS_SAMPLE_ENV, str(DEFAULT_PROCFS_SAMPLE_SEC)))
        except ValueError:
            self.procfs_sample_sec = DEFAULT_PROCFS_SAMPLE_SEC

    @staticmethod
    def _fmt_stat(value: float | None) -> str:
        if value is None:
            return "nan"
        return fmt_float(value)

    @staticmethod
    def _fmt_ms_cell(value: float | None) -> str:
        return f"{(value if value is not None else float('nan')):8.2f}"

    @staticmethod
    def _kb_to_mb(kb: float | None) -> str:
        if kb is None:
            return "nan"
        return fmt_float(kb / 1024.0)

    @staticmethod
    def _read_proc_status_kb(status_file: Path) -> int | None:
        hwm: int | None = None
        rss: int | None = None
        try:
            lines = status_file.read_text(encoding=DEFAULT_UTF8_ENCODING, errors="replace").splitlines()
        except OSError:
            return None

        for line in lines:
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    hwm = int(parts[1])
            elif line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    rss = int(parts[1])

        return hwm if hwm is not None else rss

    def _run_with_metrics(self, cmd: list[str], *, cwd: Path, log_file: Path, mem_file: Path) -> tuple[int, float | None]:
        if self.time_bin is not None and self.time_flavor == "gnu":
            with log_file.open("w", encoding=DEFAULT_UTF8_ENCODING) as handle:
                proc = subprocess.run(
                    [self.time_bin, "-f", "%M", "-o", str(mem_file), *cmd],
                    cwd=str(cwd),
                    env={**environ, "LC_ALL": "C"},
                    text=True,
                    encoding=DEFAULT_UTF8_ENCODING,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            mem_kb: float | None = None
            try:
                for line in mem_file.read_text(encoding=DEFAULT_UTF8_ENCODING).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        mem_kb = float(line)
                    except ValueError:
                        continue
            except OSError:
                mem_kb = None

            return proc.returncode, mem_kb

        if self.time_bin is not None and self.time_flavor == "bsd":
            with log_file.open("w", encoding=DEFAULT_UTF8_ENCODING) as handle:
                proc = subprocess.run(
                    [self.time_bin, "-l", "-o", str(mem_file), *cmd],
                    cwd=str(cwd),
                    env={**environ, "LC_ALL": "C"},
                    text=True,
                    encoding=DEFAULT_UTF8_ENCODING,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            mem_kb = None
            try:
                match = BSD_TIME_RSS_RE.search(mem_file.read_text(encoding=DEFAULT_UTF8_ENCODING))
                if match:
                    mem_kb = float(match.group(1)) / 1024.0   # BSD reports bytes
            except OSError:
                mem_kb = None

            return proc.returncode, mem_kb

        if self.have_procfs:
            _max_poll_sec = 3600.0  # wall-clock safety net for a hanging binary
            with log_file.open("w", encoding=DEFAULT_UTF8_ENCODING) as handle:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env={**environ, "LC_ALL": "C"},
                    text=True,
                    encoding=DEFAULT_UTF8_ENCODING,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                status_file = PROC_STATUS_ROOT / str(proc.pid) / "status"
                peak_kb = 0
                poll_start = time.perf_counter()

                while proc.poll() is None:
                    if time.perf_counter() - poll_start > _max_poll_sec:
                        proc.kill()
                        proc.wait()
                        raise RuntimeError(
                            f"benchmark process exceeded {_max_poll_sec}s wall-clock timeout"
                        )
                    sample = self._read_proc_status_kb(status_file)
                    if sample is not None and sample > peak_kb:
                        peak_kb = sample
                    time.sleep(self.procfs_sample_sec)

                sample = self._read_proc_status_kb(status_file)
                if sample is not None and sample > peak_kb:
                    peak_kb = sample

                code = proc.wait(timeout=_max_poll_sec)
                return code, float(peak_kb)

        with log_file.open("w", encoding=DEFAULT_UTF8_ENCODING) as handle:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env={**environ, "LC_ALL": "C"},
                text=True,
                encoding=DEFAULT_UTF8_ENCODING,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=3600,
            )
        return proc.returncode, None

    def _append_run_header(
        self,
        lines: list[str],
        *,
        mode: str,
        binary: Path,
        input_wav: Path,
        runs: int,
        cache_dir: Path | None,
        reset_cache: bool,
    ) -> None:
        lines.append(f"Benchmark mode: {mode}")
        lines.append(f"Binary: {binary}")
        lines.append(f"Input:  {input_wav}")
        lines.append(f"Runs:   {runs}")
        if mode == BenchMode.CACHE.value and cache_dir is not None:
            lines.append(f"Cache:  {cache_dir}")
            lines.append(f"Reset:  {'yes' if reset_cache else 'no'}")
        lines.append("")

    def _collect_cache_run(
        self,
        *,
        run_index: int,
        log_text: str,
        run_rss_kb: float | None,
        data: _RunAccumulator,
        lines: list[str],
    ) -> None:
        parsed = self.parser.parse_cache_timing(log_text)
        if parsed is None:
            log_tail = tail_lines(log_text, max_lines=DEFAULT_LONG_TAIL_LINES)
            raise RuntimeError(f"Run {run_index}: failed to parse cache timing output.\n{log_tail}")

        synth_kernel_ms, norm_ms, seg_total_ms, total_ms = parsed
        synth_wall_ms = self.parser.parse_stage_marker_medians_ms(log_text).get("synth")

        data.totals.append(total_ms)
        data.synth_kernels.append(synth_kernel_ms)
        data.norms.append(norm_ms)
        data.segbin_totals.append(seg_total_ms)
        if synth_wall_ms is not None:
            data.synth_walls.append(synth_wall_ms)

        if run_rss_kb is not None:
            data.rss_kb.append(run_rss_kb)
        run_rss_mb = self._kb_to_mb(run_rss_kb)

        if synth_wall_ms is not None:
            lines.append(
                f"run {run_index:02d}  total={total_ms:8.2f}ms  segbin={seg_total_ms:8.2f}ms  "
                f"synth_kernel={synth_kernel_ms:8.2f}ms  synth_wall={synth_wall_ms:8.2f}ms  "
                f"norm={norm_ms:8.2f}ms  rss={run_rss_mb:>8}MB"
            )
        else:
            lines.append(
                f"run {run_index:02d}  total={total_ms:8.2f}ms  segbin={seg_total_ms:8.2f}ms  "
                f"synth_kernel={synth_kernel_ms:8.2f}ms  norm={norm_ms:8.2f}ms  rss={run_rss_mb:>8}MB"
            )

    def _collect_normal_run(
        self,
        *,
        run_index: int,
        log_text: str,
        run_rss_kb: float | None,
        data: _RunAccumulator,
        lines: list[str],
    ) -> None:
        parsed = self.parser.parse_normal_timing(log_text)
        if parsed is None:
            log_tail = tail_lines(log_text, max_lines=DEFAULT_LONG_TAIL_LINES)
            raise RuntimeError(f"Run {run_index}: failed to parse normal timing output.\n{log_tail}")

        fft_ms, track_ms, synth_kernel_ms, norm_ms, total_ms = parsed
        synth_wall_ms = self.parser.parse_stage_marker_medians_ms(log_text).get("synth")

        data.totals.append(total_ms)
        data.ffts.append(fft_ms)
        data.tracks.append(track_ms)
        data.synth_kernels.append(synth_kernel_ms)
        data.norms.append(norm_ms)
        if synth_wall_ms is not None:
            data.synth_walls.append(synth_wall_ms)

        fft_bw_gibps, track_bw_gibps = self.parser.parse_analysis_bandwidth(log_text)
        fft_bw_rel_pct: float | None = None
        track_bw_rel_pct: float | None = None

        if fft_bw_gibps is not None:
            data.fft_bandwidths.append(fft_bw_gibps)
        if track_bw_gibps is not None:
            data.track_bandwidths.append(track_bw_gibps)

        if fft_bw_gibps is not None and track_bw_gibps is not None:
            faster_stage_bw = max(fft_bw_gibps, track_bw_gibps)
            if faster_stage_bw > 0.0:
                fft_bw_rel_pct = 100.0 * fft_bw_gibps / faster_stage_bw
                track_bw_rel_pct = 100.0 * track_bw_gibps / faster_stage_bw
                data.fft_bandwidth_rel_pcts.append(fft_bw_rel_pct)
                data.track_bandwidth_rel_pcts.append(track_bw_rel_pct)

        if run_rss_kb is not None:
            data.rss_kb.append(run_rss_kb)
        run_rss_mb = self._kb_to_mb(run_rss_kb)

        synth_wall_cell = self._fmt_ms_cell(synth_wall_ms)
        if (
            fft_bw_gibps is not None
            and track_bw_gibps is not None
            and fft_bw_rel_pct is not None
            and track_bw_rel_pct is not None
        ):
            lines.append(
                f"run {run_index:02d}  total={total_ms:8.2f}ms  fft={fft_ms:8.2f}ms  "
                f"track={track_ms:8.2f}ms  synth_kernel={synth_kernel_ms:8.2f}ms  "
                f"synth_wall={synth_wall_cell}ms  norm={norm_ms:8.2f}ms  "
                f"rss={run_rss_mb:>8}MB  fft_bw={fft_bw_gibps:6.2f}GiB/s "
                f"({fft_bw_rel_pct:5.1f}% of faster stage)  "
                f"track_bw={track_bw_gibps:6.2f}GiB/s ({track_bw_rel_pct:5.1f}% of faster stage)"
            )
        else:
            lines.append(
                f"run {run_index:02d}  total={total_ms:8.2f}ms  fft={fft_ms:8.2f}ms  "
                f"track={track_ms:8.2f}ms  synth_kernel={synth_kernel_ms:8.2f}ms  "
                f"synth_wall={synth_wall_cell}ms  norm={norm_ms:8.2f}ms  "
                f"rss={run_rss_mb:>8}MB"
            )

    def run(
        self,
        *,
        binary: Path,
        input_wav: Path,
        runs: int,
        mode: str,
        cli_args: list[str],
        cache_dir: Path | None,
        reset_cache: bool,
    ) -> BenchRunResult:
        if runs < 1:
            raise RuntimeError("--runs must be a positive integer")
        mode_values = {mode_item.value for mode_item in BenchMode}
        if mode not in mode_values:
            raise RuntimeError("--mode must be 'normal' or 'cache'")

        binary = binary.resolve()
        input_wav = input_wav.resolve()

        if not is_executable(binary):
            raise RuntimeError(f"binary not executable: {binary}")
        if not input_wav.is_file():
            raise RuntimeError(f"input file not found: {input_wav}")

        if self.dry_run:
            plan = [str(binary), str(input_wav), *cli_args]
            if mode == BenchMode.CACHE.value:
                plan.append(CACHE_FLAG)
            text = "\n".join(
                [
                    f"Benchmark mode: {mode}",
                    f"Binary: {binary}",
                    f"Input:  {input_wav}",
                    f"Runs:   {runs}",
                    "",
                    "[dry-run] planned command:",
                    "  " + " ".join(shlex.quote(part) for part in plan),
                ]
            )
            return BenchRunResult(
                stdout=text + "\n",
                metrics={key: {} for key in METRIC_SECTION_KEYS},
                command={"cmd": plan, "cwd": str(self.repo_root), "returncode": 0, "total_elapsed_sec": 0.0},
            )

        lines: list[str] = []
        data = _RunAccumulator()

        with tempfile.TemporaryDirectory(prefix="spectral-bench.") as tmpdir_raw:
            tmpdir = Path(tmpdir_raw)
            run_dir = tmpdir / "run"
            output_dir = run_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            effective_cache_dir: Path | None = None
            if mode == BenchMode.CACHE.value:
                effective_cache_dir = (tmpdir / "cache") if cache_dir is None else cache_dir.resolve()
                effective_cache_dir.mkdir(parents=True, exist_ok=True)

                cache_link = output_dir / "cache"
                if cache_link.exists() or cache_link.is_symlink():
                    cache_link.unlink()
                cache_link.symlink_to(effective_cache_dir)

                if reset_cache:
                    for filename in CACHE_FILES:
                        path = effective_cache_dir / filename
                        if path.exists():
                            path.unlink()

            self._append_run_header(
                lines,
                mode=mode,
                binary=binary,
                input_wav=input_wav,
                runs=runs,
                cache_dir=effective_cache_dir,
                reset_cache=reset_cache,
            )

            start = time.perf_counter()
            for run_index in range(1, runs + 1):
                log_file = tmpdir / f"run_{run_index}.log"
                mem_file = tmpdir / f"run_{run_index}.rss"
                cmd = [str(binary), str(input_wav), *cli_args]
                if mode == BenchMode.CACHE.value:
                    cmd.append(CACHE_FLAG)

                code, run_rss_kb = self._run_with_metrics(cmd, cwd=run_dir, log_file=log_file, mem_file=mem_file)
                if code != 0:
                    log_tail = tail_lines(
                        log_file.read_text(encoding=DEFAULT_UTF8_ENCODING, errors="replace"),
                        max_lines=DEFAULT_LONG_TAIL_LINES,
                    )
                    raise RuntimeError(f"Run {run_index} failed. Last log lines:\n{log_tail}")

                log_text = log_file.read_text(encoding=DEFAULT_UTF8_ENCODING, errors="replace")
                if mode == BenchMode.CACHE.value:
                    self._collect_cache_run(
                        run_index=run_index,
                        log_text=log_text,
                        run_rss_kb=run_rss_kb,
                        data=data,
                        lines=lines,
                    )
                else:
                    self._collect_normal_run(
                        run_index=run_index,
                        log_text=log_text,
                        run_rss_kb=run_rss_kb,
                        data=data,
                        lines=lines,
                    )

                if data.first_total is None:
                    data.first_total = data.totals[-1]

            elapsed = time.perf_counter() - start

        all_median = self.parser.median(data.totals)
        all_mean = self.parser.mean(data.totals)

        warm_totals = data.totals[1:] if len(data.totals) > 1 else []
        warm_median = self.parser.median(warm_totals)
        warm_mean = self.parser.mean(warm_totals)

        lines.append("")
        lines.append("--- Summary ---")
        lines.append(
            "Total ms: "
            f"first={(data.first_total or 0.0):.3f} "
            f"median={self._fmt_stat(all_median)} "
            f"mean={self._fmt_stat(all_mean)} "
            f"warm_median={self._fmt_stat(warm_median)} "
            f"warm_mean={self._fmt_stat(warm_mean)}"
        )

        if mode == BenchMode.CACHE.value:
            seg_median = self.parser.median(data.segbin_totals)
            synth_kernel_median = self.parser.median(data.synth_kernels)
            synth_wall_median = self.parser.median(data.synth_walls)
            norm_median = self.parser.median(data.norms)
            lines.append(
                "Segment-binary ms: "
                f"total_median={self._fmt_stat(seg_median)} "
                f"synth_kernel_median={self._fmt_stat(synth_kernel_median)} "
                f"synth_wall_median={self._fmt_stat(synth_wall_median)} "
                f"norm_median={self._fmt_stat(norm_median)}"
            )
        else:
            fft_median = self.parser.median(data.ffts)
            track_median = self.parser.median(data.tracks)
            synth_kernel_median = self.parser.median(data.synth_kernels)
            synth_wall_median = self.parser.median(data.synth_walls)
            norm_median = self.parser.median(data.norms)
            lines.append(
                "Stage medians ms: "
                f"fft={self._fmt_stat(fft_median)} "
                f"track={self._fmt_stat(track_median)} "
                f"synth_kernel={self._fmt_stat(synth_kernel_median)} "
                f"synth_wall={self._fmt_stat(synth_wall_median)} "
                f"norm={self._fmt_stat(norm_median)}"
            )

            if (
                data.fft_bandwidths
                and data.track_bandwidths
                and data.fft_bandwidth_rel_pcts
                and data.track_bandwidth_rel_pcts
            ):
                fft_bw_median = self.parser.median(data.fft_bandwidths)
                fft_bw_rel_pct_median = self.parser.median(data.fft_bandwidth_rel_pcts)
                track_bw_median = self.parser.median(data.track_bandwidths)
                track_bw_rel_pct_median = self.parser.median(data.track_bandwidth_rel_pcts)
                lines.append(
                    "Stage bandwidth medians: "
                    f"fft={self._fmt_stat(fft_bw_median)}GiB/s "
                    f"({self._fmt_stat(fft_bw_rel_pct_median)}% of faster stage) "
                    f"track={self._fmt_stat(track_bw_median)}GiB/s "
                    f"({self._fmt_stat(track_bw_rel_pct_median)}% of faster stage)"
                )
                lines.append(
                    "Bandwidth percentages are relative between FFT and track "
                    "(not hardware peak utilization)."
                )

        if data.rss_kb:
            rss_median_kb = self.parser.median(data.rss_kb)
            rss_mean_kb = self.parser.mean(data.rss_kb)
            warm_rss = data.rss_kb[1:] if len(data.rss_kb) > 1 else []
            warm_rss_median_kb = self.parser.median(warm_rss)
            warm_rss_mean_kb = self.parser.mean(warm_rss)

            lines.append(
                "Memory max RSS (MB): "
                f"median={self._kb_to_mb(rss_median_kb)} "
                f"mean={self._kb_to_mb(rss_mean_kb)} "
                f"warm_median={self._kb_to_mb(warm_rss_median_kb)} "
                f"warm_mean={self._kb_to_mb(warm_rss_mean_kb)}"
            )
        else:
            lines.append("Memory max RSS: unavailable (no GNU time and no /proc sampler)")

        stdout = "\n".join(lines) + "\n"
        metrics = self.parser.parse_benchmark_output(stdout)

        return BenchRunResult(
            stdout=stdout,
            metrics=metrics,
            command={
                "cmd": [str(binary), str(input_wav), *cli_args],
                "cwd": str(self.repo_root),
                "returncode": 0,
                "total_elapsed_sec": elapsed,
            },
        )
