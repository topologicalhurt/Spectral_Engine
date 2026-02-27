"""Core git-layer models shared by git tooling modules."""

from __future__ import annotations

from dataclasses import dataclass

GIT_DEFAULT_TIMEOUT_SEC = 15


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RefLookupResult:
    ref: str | None
    note: str = ""
