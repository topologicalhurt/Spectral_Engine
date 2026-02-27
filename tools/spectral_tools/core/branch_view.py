"""Branch-style report formatting shared across tool scripts."""

from __future__ import annotations

import math
import statistics
from typing import Any

from .console import Console
from .constants import DEFAULT_LONG_TEXT_CHARS, DEFAULT_TEXT_CLIP_CHARS
from .utils import fmt_float


class BranchFormatter:
    HIDDEN_KEYS = {"required_contract"}
    PERF_STATUS_ORDER = ("ok", "failed", "blocked", "unavailable", "skipped")

    def __init__(self, use_color: bool = True) -> None:
        self.console = Console(use_color=use_color)

    @staticmethod
    def _clip_text(value: str, max_chars: int = DEFAULT_TEXT_CLIP_CHARS) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."

    @staticmethod
    def _extract_float(dct: dict[str, Any], path: tuple[str, ...]) -> float | None:
        cur: Any = dct
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        if isinstance(cur, int):
            return float(cur)
        if isinstance(cur, float):
            if not math.isfinite(cur):
                return None
            return float(cur)
        return None

    def _scalar_text(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if not math.isfinite(value):
                return "null"
            return fmt_float(value)
        if isinstance(value, str):
            if value == "":
                return "\"\""
            if "\n" in value:
                lines = value.splitlines()
                first = self._clip_text(lines[0].strip() if lines else "")
                last = self._clip_text(lines[-1].strip() if lines else "")
                if len(lines) > 1:
                    peek = f"peek='{first}' ... '{last}'"
                else:
                    peek = f"peek='{first}'"
                return (
                    f"<multiline text: {len(lines)} lines, {len(value)} chars"
                    f"; {peek}; use -r/--raw for full content>"
                )
            if len(value) > DEFAULT_LONG_TEXT_CHARS:
                return value[: DEFAULT_LONG_TEXT_CHARS - 3] + "..."
        return str(value)

    def _status_text(self, raw: str) -> str:
        return self.console.status((raw or "").strip())

    def _value_text(self, key: str, value: Any) -> str:
        if key == "status" and isinstance(value, str):
            return self._status_text(value)
        return self._scalar_text(value)

    def _render_branches(self, key: str, value: Any, prefix: str, is_last: bool, out: list[str]) -> None:
        if key in self.HIDDEN_KEYS:
            return

        connector = "`- " if is_last else "|- "
        next_prefix = prefix + ("   " if is_last else "|  ")

        if isinstance(value, dict):
            items = [(k, v) for k, v in value.items() if str(k) not in self.HIDDEN_KEYS]
            if not items:
                return
            out.append(f"{prefix}{connector}{key}")
            for idx, (child_key, child_val) in enumerate(items):
                self._render_branches(str(child_key), child_val, next_prefix, idx == len(items) - 1, out)
            return

        if isinstance(value, list):
            out.append(f"{prefix}{connector}{key}")
            for idx, child_val in enumerate(value):
                self._render_branches(f"[{idx}]", child_val, next_prefix, idx == len(value) - 1, out)
            return

        out.append(f"{prefix}{connector}{key}: {self._value_text(key, value)}")

    def _metric_stats_line(self, label: str, values: list[float]) -> str:
        if not values:
            return f"{label}: n=0"
        ordered_vals = sorted(float(value) for value in values)
        min_val = ordered_vals[0]
        median_val = float(statistics.median(ordered_vals))
        mean_val = float(statistics.fmean(ordered_vals))
        max_val = ordered_vals[-1]

        return (
            f"{label}: n={len(values)} "
            f"min={fmt_float(min_val)} "
            f"median={fmt_float(median_val)} "
            f"mean={fmt_float(mean_val)} "
            f"max={fmt_float(max_val)}"
        )

    def _aggregate_lines(self, tests: list[Any]) -> tuple[list[str], int, int, int]:
        """Return (lines, ok, fail, skip) from test list."""
        ok_count = 0
        fail_count = 0
        skip_count = 0

        perf_statuses: dict[str, int] = {}
        total_medians: list[float] = []
        track_medians: list[float] = []
        synth_kernel_medians: list[float] = []
        synth_wall_medians: list[float] = []

        for test in tests:
            if not isinstance(test, dict):
                continue

            status_norm = str(test.get("status", "unknown")).strip().lower()
            if status_norm == "ok":
                ok_count += 1
            elif status_norm == "failed":
                fail_count += 1
            elif status_norm == "skipped":
                skip_count += 1

            details = test.get("details")
            if not isinstance(details, dict):
                continue

            metrics = details.get("metrics")
            have_synth_wall = False
            if isinstance(metrics, dict):
                total_median = self._extract_float(metrics, ("summary", "median_ms"))
                track_median = self._extract_float(metrics, ("track_series", "median_ms"))
                synth_kernel = self._extract_float(metrics, ("stage_medians", "synth_kernel_ms"))
                if synth_kernel is None:
                    synth_kernel = self._extract_float(metrics, ("stage_medians", "synth_ms"))
                synth_wall = self._extract_float(metrics, ("stage_medians", "synth_wall_ms"))

                if total_median is not None:
                    total_medians.append(total_median)
                if track_median is not None:
                    track_medians.append(track_median)
                if synth_kernel is not None:
                    synth_kernel_medians.append(synth_kernel)
                if synth_wall is not None:
                    synth_wall_medians.append(synth_wall)
                    have_synth_wall = True

            perf_profile = details.get("perf_profile")
            if isinstance(perf_profile, dict):
                perf_status = str(perf_profile.get("status", "unknown")).strip().lower() or "unknown"
                perf_statuses[perf_status] = perf_statuses.get(perf_status, 0) + 1
                if not have_synth_wall:
                    perf_synth_wall = self._extract_float(
                        perf_profile,
                        ("marker_summary", "stage_duration_stats_ms", "synth", "median_ms"),
                    )
                    if perf_synth_wall is not None:
                        synth_wall_medians.append(perf_synth_wall)

        lines: list[str] = []
        lines.append(
            f"tests: total={len(tests)} "
            f"ok={self.console.green(str(ok_count))} "
            f"failed={self.console.red(str(fail_count))} "
            f"skipped={self.console.yellow(str(skip_count))}"
        )

        if perf_statuses:
            status_parts: list[str] = []
            for label in self.PERF_STATUS_ORDER:
                if label in perf_statuses:
                    status_parts.append(f"{self._status_text(label)}={perf_statuses[label]}")
            for label in perf_statuses.keys():
                if label not in self.PERF_STATUS_ORDER:
                    count = perf_statuses[label]
                    status_parts.append(f"{label}={count}")
            lines.append(f"{self.console.cyan('-P')}: " + " ".join(status_parts))

        if total_medians:
            lines.append(self._metric_stats_line("total_median_ms", total_medians))
        if track_medians:
            lines.append(self._metric_stats_line("track_median_ms", track_medians))
        if synth_kernel_medians:
            lines.append(self._metric_stats_line("synth_kernel_median_ms", synth_kernel_medians))
        if synth_wall_medians:
            lines.append(self._metric_stats_line("synth_wall_median_ms", synth_wall_medians))

        return lines, ok_count, fail_count, skip_count

    def render(self, report: dict[str, Any]) -> str:
        lines: list[str] = []

        suite_name = str(report.get("suite", "report"))
        lines.append(self.console.bold(f"Performance Suite: {suite_name}"))

        context = report.get("context")
        if isinstance(context, dict) and context:
            lines.append("")
            lines.append("Context")
            items = list(context.items())
            for idx, (key, value) in enumerate(items):
                if isinstance(value, (dict, list)):
                    self._render_branches(str(key), value, "", idx == len(items) - 1, lines)
                else:
                    lines.append(f"{key}: {self._scalar_text(value)}")

        tests = report.get("tests")
        if not isinstance(tests, list):
            tests = []

        agg_lines, ok_count, fail_count, skip_count = self._aggregate_lines(tests)
        lines.extend(agg_lines)

        lines.append("")
        lines.append("Results")
        for test in tests:
            if not isinstance(test, dict):
                continue

            name = str(test.get("name", "unnamed"))
            status_raw = str(test.get("status", "unknown"))
            summary = str(test.get("summary", "")).strip()
            status_view = self._status_text(status_raw)

            details = test.get("details")
            perf_tag = ""
            if isinstance(details, dict):
                perf_profile = details.get("perf_profile")
                if isinstance(perf_profile, dict):
                    perf_state = str(perf_profile.get("status", "unknown"))
                    perf_tag = f"  {self.console.cyan('-P')}={self._status_text(perf_state)}"

            if summary:
                lines.append(f"  {name:<34} {status_view}{perf_tag}  {summary}")
            else:
                lines.append(f"  {name:<34} {status_view}{perf_tag}")

            if isinstance(details, dict) and details:
                detail_items = list(details.items())
                for idx, (key, value) in enumerate(detail_items):
                    self._render_branches(str(key), value, "    ", idx == len(detail_items) - 1, lines)

        lines.append("")
        lines.append(
            f"{self.console.green(str(ok_count))} succeeded, "
            f"{self.console.red(str(fail_count))} failed, "
            f"{self.console.yellow(str(skip_count))} skipped"
        )

        return "\n".join(lines)
