"""Shared tooling utilities for scripts under tools/.

Eagerly imports lightweight helpers (utils, constants).
Heavier modules (benchmark_*, perf_profile, branch_view, resource_hashes,
subtree_*) are lazily loaded via __getattr__ to keep ``import spectral_tools``
fast and avoid pulling in numpy/xxhash at startup.

The __all__ list covers the public API surface for external callers such as
entry-point scripts.  Module-internal helpers (e.g. Performance, BenchmarkRunner)
are intentionally excluded — callers import them directly from their submodule.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .constants import (
    DEFAULT_BENCH_MODE,
    DEFAULT_BENCH_RUNS,
    DEFAULT_SUITE_RUNS,
)
from .utils import (
    find_repo_root,
    fmt_float,
    is_executable,
    list_repo_files_or_rglob,
    parse_int_csv,
    read_utf8_lf,
    tail_lines,
    write_utf8_lf,
)

_LAZY_EXPORTS = {
    "BranchFormatter": (".branch_view", "BranchFormatter"),
    "Console": (".console", "Console"),
    "DEFAULT_PERF_EVENTS": (".perf_profile", "DEFAULT_PERF_EVENTS"),
    "profile_stage_markers_contract": (".perf_profile", "profile_stage_markers_contract"),
    "profile_stage_matrix": (".perf_profile", "profile_stage_matrix"),
    "ProcessError": (".process", "ProcessError"),
    "ProcessResult": (".process", "ProcessResult"),
    "run": (".process", "run"),
    "render_report": (".report_output", "render_report"),
    "serialize_report": (".report_output", "serialize_report"),
    "write_report_json": (".report_output", "write_report_json"),
}

__all__ = [
    "BranchFormatter",
    "Console",
    "DEFAULT_BENCH_MODE",
    "DEFAULT_BENCH_RUNS",
    "DEFAULT_PERF_EVENTS",
    "DEFAULT_SUITE_RUNS",
    "ProcessError",
    "ProcessResult",
    "find_repo_root",
    "fmt_float",
    "is_executable",
    "list_repo_files_or_rglob",
    "parse_int_csv",
    "profile_stage_markers_contract",
    "profile_stage_matrix",
    "read_utf8_lf",
    "render_report",
    "run",
    "serialize_report",
    "tail_lines",
    "write_utf8_lf",
    "write_report_json",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
