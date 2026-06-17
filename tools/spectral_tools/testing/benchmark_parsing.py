"""Parsing and metrics helpers for benchmark output."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from ..core.constants import STAGE_MARKER_LINE_RE

RUN_TRACK_RE = re.compile(r"^run\s+\d+.*?track=\s*([0-9]+(?:\.[0-9]+)?)ms", re.MULTILINE)
RUN_TRACK_BW_RE = re.compile(r"^run\s+\d+.*?track_bw=\s*([0-9]+(?:\.[0-9]+)?)GiB/s", re.MULTILINE)
TOTAL_RE = re.compile(
    r"^Total ms: first=(\S+) median=(\S+) mean=(\S+) warm_median=(\S+) warm_mean=(\S+)",
    re.MULTILINE,
)
# "Stage medians ms:" is the legacy (cold-run-inclusive) label; "Stage warm
# medians ms:" is the current steady-state label. The optional "(?:warm )?"
# accepts both so a freshly-emitted log and an archived report both parse.
STAGE_RE_V2 = re.compile(
    r"^Stage (?:warm )?medians ms: fft=(\S+) track=(\S+) synth_kernel=(\S+) synth_wall=(\S+) norm=(\S+)",
    re.MULTILINE,
)
STAGE_RE_V1 = re.compile(
    r"^Stage (?:warm )?medians ms: fft=(\S+) track=(\S+) synth=(\S+) norm=(\S+)", re.MULTILINE
)
STAGE_SPREAD_RE = re.compile(r"^Stage warm spread ms \(min\.\.max\): (.+)$", re.MULTILINE)
SPREAD_TOKEN_RE = re.compile(r"\b(fft|track|synth_kernel|synth_wall|norm)=([0-9.]+|nan)\.\.([0-9.]+|nan)")
BW_RE = re.compile(
    r"^Stage bandwidth (?:warm )?medians: fft=(\S+)GiB/s \(([^)]+)% of faster stage\) "
    r"track=(\S+)GiB/s \(([^)]+)% of faster stage\)",
    re.MULTILINE,
)
RSS_RE = re.compile(
    r"^Memory max RSS \(MB\): median=(\S+) mean=(\S+) warm_median=(\S+) warm_mean=(\S+)",
    re.MULTILINE,
)
CACHE_SEG_TIMING_RE = re.compile(
    r"Segment-binary synth run:.*?"
    r"Synth\s*([0-9]+(?:\.[0-9]+)?)ms.*?"
    r"Norm\s*([0-9]+(?:\.[0-9]+)?)ms.*?"
    r"Total\s*([0-9]+(?:\.[0-9]+)?)ms"
)
CACHE_RENDER_RE = re.compile(
    r"Render:.*?"
    r"Synth\s*([0-9]+(?:\.[0-9]+)?)ms.*?"
    r"Norm\s*([0-9]+(?:\.[0-9]+)?)ms.*?"
    r"\(([0-9]+(?:\.[0-9]+)?)ms\)"
)
CACHE_TOTAL_RE = re.compile(r"Cache-mode end-to-end total:\s*([0-9]+(?:\.[0-9]+)?)ms")
BW_TOKEN_RE = re.compile(r"\bbw=([0-9]+(?:\.[0-9]+)?)GiB/s")
NAN_TOKEN = "nan"
NORMAL_TIMING_TOKENS = {"FFT:": "fft", "Track:": "track", "Synth:": "synth", "Norm:": "norm", "Total:": "total"}


class BenchmarkOutputParser:
    """Parses benchmark log output and computes derived summary stats."""

    @staticmethod
    def to_float(token: str) -> float | None:
        if token.lower() == NAN_TOKEN:
            return None
        try:
            return float(token)
        except ValueError:
            return None

    @staticmethod
    def median(values: list[float]) -> float | None:
        if not values:
            return None
        return float(np.median(np.asarray(values, dtype=float)))

    @staticmethod
    def mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(np.mean(np.asarray(values, dtype=float)))

    @staticmethod
    def stdev(values: list[float]) -> float | None:
        if not values:
            return None
        return float(np.std(np.asarray(values, dtype=float), ddof=0))

    @staticmethod
    def percentile_linear(sorted_values: list[float], percentile: float) -> float:
        if not sorted_values:
            raise ValueError("empty values")
        return float(np.quantile(np.asarray(sorted_values, dtype=float), percentile, method="linear"))

    @classmethod
    def iqr_outlier_count(cls, values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        ordered = sorted(values)
        q1 = cls.percentile_linear(ordered, 0.25)
        q3 = cls.percentile_linear(ordered, 0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        count = sum(1 for v in ordered if v < lo or v > hi)
        return {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "low_fence": lo,
            "high_fence": hi,
            "count": count,
        }

    @staticmethod
    def parse_normal_timing(log_text: str) -> tuple[float, float, float, float, float] | None:
        last: tuple[float, float, float, float, float] | None = None
        for line in log_text.splitlines():
            if not line.startswith("FFT:"):
                continue

            values: dict[str, float] = {}
            tokens = line.split()
            for idx, token in enumerate(tokens):
                key = NORMAL_TIMING_TOKENS.get(token)
                if key is None:
                    continue
                if idx + 1 >= len(tokens):
                    continue
                raw = tokens[idx + 1]
                if raw.endswith("ms"):
                    raw = raw[:-2]
                try:
                    values[key] = float(raw)
                except ValueError:
                    continue

            required = ("fft", "track", "synth", "norm", "total")
            if all(name in values for name in required):
                last = (
                    values["fft"],
                    values["track"],
                    values["synth"],
                    values["norm"],
                    values["total"],
                )
        return last

    @staticmethod
    def parse_cache_timing(log_text: str) -> tuple[float, float, float, float] | None:
        synth: float | None = None
        norm: float | None = None
        seg_total: float | None = None
        end_to_end: float | None = None

        for line in log_text.splitlines():
            seg_match = CACHE_SEG_TIMING_RE.search(line)
            if seg_match:
                synth, norm, seg_total = (float(token) for token in seg_match.groups())
                continue

            render_match = CACHE_RENDER_RE.search(line)
            if render_match:
                synth, norm, seg_total = (float(token) for token in render_match.groups())
                continue

            total_match = CACHE_TOTAL_RE.search(line)
            if total_match:
                end_to_end = float(total_match.group(1))

        if synth is None or norm is None or seg_total is None or end_to_end is None:
            return None
        return synth, norm, seg_total, end_to_end

    @staticmethod
    def parse_analysis_bandwidth(log_text: str) -> tuple[float | None, float | None]:
        fft_bw: float | None = None
        track_bw: float | None = None

        for line in log_text.splitlines():
            if line.startswith("Analysis") and "FFT:" in line:
                match = BW_TOKEN_RE.search(line)
                if match:
                    fft_bw = float(match.group(1))
                continue

            if line.startswith("Analysis track:"):
                match = BW_TOKEN_RE.search(line)
                if match:
                    track_bw = float(match.group(1))

        return fft_bw, track_bw

    @staticmethod
    def parse_stage_marker_medians_ms(log_text: str) -> dict[str, float]:
        starts: dict[str, list[int]] = {}
        spans_ms: dict[str, list[float]] = {}

        for line in log_text.splitlines():
            match = STAGE_MARKER_LINE_RE.match(line.strip())
            if not match:
                continue
            kind, stage, mono_ns_raw = match.groups()
            mono_ns = int(mono_ns_raw)

            if kind == "BEGIN":
                starts.setdefault(stage, []).append(mono_ns)
                continue

            stage_starts = starts.get(stage)
            if stage_starts:
                start_ns = stage_starts.pop()
                delta_ns = mono_ns - start_ns
                if delta_ns < 0:
                    delta_ns = 0
                spans_ms.setdefault(stage, []).append(delta_ns / 1_000_000.0)

        medians: dict[str, float] = {}
        for stage, values in spans_ms.items():
            if values:
                medians[stage] = float(np.median(np.asarray(values, dtype=float)))
        return medians

    def parse_benchmark_output(self, output: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "summary": {},
            "stage_medians": {},
            "stage_spread": {},
            "bandwidth_medians": {},
            "memory": {},
            "track_series": {},
        }

        total_match = TOTAL_RE.search(output)
        if total_match:
            first, median, mean, warm_median, warm_mean = total_match.groups()
            metrics["summary"] = {
                "first_ms": self.to_float(first),
                "median_ms": self.to_float(median),
                "mean_ms": self.to_float(mean),
                "warm_median_ms": self.to_float(warm_median),
                "warm_mean_ms": self.to_float(warm_mean),
            }

        stage_match_v2 = STAGE_RE_V2.search(output)
        if stage_match_v2:
            fft, track, synth_kernel, synth_wall, norm = stage_match_v2.groups()
            metrics["stage_medians"] = {
                "fft_ms": self.to_float(fft),
                "track_ms": self.to_float(track),
                "synth_kernel_ms": self.to_float(synth_kernel),
                "synth_wall_ms": self.to_float(synth_wall),
                "synth_ms": self.to_float(synth_kernel),
                "norm_ms": self.to_float(norm),
            }
        else:
            stage_match_v1 = STAGE_RE_V1.search(output)
            if stage_match_v1:
                fft, track, synth, norm = stage_match_v1.groups()
                metrics["stage_medians"] = {
                    "fft_ms": self.to_float(fft),
                    "track_ms": self.to_float(track),
                    "synth_kernel_ms": self.to_float(synth),
                    "synth_ms": self.to_float(synth),
                    "norm_ms": self.to_float(norm),
                }

        spread_match = STAGE_SPREAD_RE.search(output)
        if spread_match:
            spread: dict[str, float | None] = {}
            for key, lo, hi in SPREAD_TOKEN_RE.findall(spread_match.group(1)):
                spread[f"{key}_min_ms"] = self.to_float(lo)
                spread[f"{key}_max_ms"] = self.to_float(hi)
            metrics["stage_spread"] = spread

        bw_match = BW_RE.search(output)
        if bw_match:
            fft_bw, fft_rel, track_bw, track_rel = bw_match.groups()
            metrics["bandwidth_medians"] = {
                "fft_gibps": self.to_float(fft_bw),
                "fft_rel_pct": self.to_float(fft_rel),
                "track_gibps": self.to_float(track_bw),
                "track_rel_pct": self.to_float(track_rel),
            }

        rss_match = RSS_RE.search(output)
        if rss_match:
            median, mean, warm_median, warm_mean = rss_match.groups()
            metrics["memory"] = {
                "rss_median_mb": self.to_float(median),
                "rss_mean_mb": self.to_float(mean),
                "rss_warm_median_mb": self.to_float(warm_median),
                "rss_warm_mean_mb": self.to_float(warm_mean),
            }

        track_values = [float(v) for v in RUN_TRACK_RE.findall(output)]
        track_bw_values = [float(v) for v in RUN_TRACK_BW_RE.findall(output)]

        warm_track_values = track_values[1:] if len(track_values) > 1 else []
        outliers = self.iqr_outlier_count(track_values)

        metrics["track_series"] = {
            "count": len(track_values),
            "median_ms": self.median(track_values),
            "warm_median_ms": self.median(warm_track_values),
            "mean_ms": self.mean(track_values),
            "stdev_ms": self.stdev(track_values),
            "min_ms": min(track_values) if track_values else None,
            "max_ms": max(track_values) if track_values else None,
            "outliers_iqr": outliers,
            "bw_median_gibps": self.median(track_bw_values),
        }
        self._warn_if_empty(metrics)
        return metrics

    def _warn_if_empty(self, metrics: dict[str, Any]) -> None:
        """Log a warning if no top-level regex matched — likely a format change."""
        if (
            not metrics["summary"]
            and not metrics["stage_medians"]
            and not metrics["bandwidth_medians"]
            and not metrics["memory"]
            and metrics["track_series"].get("count", 0) == 0
        ):
            logging.warning(
                "parse_benchmark_output: no metrics parsed from output. "
                "The benchmark log format may have changed."
            )
