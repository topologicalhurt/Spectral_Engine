"""Process helpers shared by Python tooling scripts."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any, Mapping

from .constants import DEFAULT_UTF8_ENCODING


@dataclass(slots=True)
class ProcessResult:
    cmd: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "cmd": self.cmd,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "elapsed_sec": self.elapsed_sec,
        }


class ProcessError(RuntimeError):
    pass


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_sec: int | None = None,
    check: bool = False,
) -> ProcessResult:
    merged_env = dict(environ)
    if env is not None:
        merged_env.update(env)

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            encoding=DEFAULT_UTF8_ENCODING,
            capture_output=True,
            env=merged_env,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        result = ProcessResult(
            cmd=cmd,
            cwd=str(cwd),
            returncode=-9,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            elapsed_sec=elapsed,
        )
        if check:
            raise ProcessError(
                f"command timed out after {timeout_sec}s: {' '.join(cmd)}"
            ) from exc
        return result
    elapsed = time.perf_counter() - start

    result = ProcessResult(
        cmd=cmd,
        cwd=str(cwd),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_sec=elapsed,
    )

    if check and result.returncode != 0:
        raise ProcessError(
            f"command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{result.stderr.strip()}\n"
            f"stdout:\n{result.stdout.strip()}"
        )

    return result
