"""Shared constants for tooling scripts under tools/."""

from __future__ import annotations

import re

DEFAULT_UTF8_ENCODING = "utf-8"

DEFAULT_FLOAT_DECIMALS = 3
DEFAULT_SHORT_TAIL_LINES = 12
DEFAULT_LONG_TAIL_LINES = 80
DEFAULT_PEEK_TAIL_LINES = 60

DEFAULT_BENCH_RUNS = 6
DEFAULT_SUITE_RUNS = 25
DEFAULT_BENCH_MODE = "normal"
DEFAULT_SUITE_INPUT = "resources/testing/shakespeare_he_saw_the_cat.wav"
DEFAULT_SUITE_BENCH_ARGS = "0 1 0 4096 128 -85 20 0"
DEFAULT_CHUNK_VALUES = "0,1,2,4,8,16,32"
DEFAULT_PERF_TIMEOUT_SEC = 0
DEFAULT_PROCFS_SAMPLE_SEC = 0.001

DEFAULT_TEXT_CLIP_CHARS = 96
DEFAULT_LONG_TEXT_CHARS = 160
REMAINDER_DELIMITER = "--"
NO_COLOR_ENV = "NO_COLOR"
REPO_ROOT_ENV = "SPECTRAL_REPO_ROOT"

DEFAULT_BUILD_TARGET = "desktop"
DEFAULT_BENCHMARK_BINARY_GLOB = "spectral_*_desktop"

# Stage marker protocol regex — shared by perf_profile and benchmark_parsing.
STAGE_MARKER_LINE_RE = re.compile(
    r"^SPECTRAL_STAGE_(BEGIN|END)\s+([a-zA-Z0-9_.:/-]+)\s+([0-9]+)\s*$"
)


def short_ref(value: str | None) -> str:
    """Truncate a git ref to 12 characters for display."""
    if not value:
        return "unknown"
    return value[:12]
