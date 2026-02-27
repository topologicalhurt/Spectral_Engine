"""Shared dataclasses for benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class BenchMode(str, Enum):
    NORMAL = "normal"
    CACHE = "cache"


class PerfStrategy(str, Enum):
    MATRIX = "matrix"
    MARKERS = "markers"


class TestStatus(str, Enum):
    OK = "ok"
    FAILED = "FAILED"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class BenchRunResult:
    stdout: str
    metrics: dict[str, Any]
    command: dict[str, Any]
