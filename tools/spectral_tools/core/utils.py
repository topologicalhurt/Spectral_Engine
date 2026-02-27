"""Common repository, text, and scalar helpers for tools/ scripts."""

from __future__ import annotations

import shutil

from itertools import chain
from os import environ
from pathlib import Path
from stat import S_IXGRP, S_IXOTH, S_IXUSR

from .constants import (
    DEFAULT_FLOAT_DECIMALS,
    DEFAULT_SHORT_TAIL_LINES,
    DEFAULT_UTF8_ENCODING,
    REPO_ROOT_ENV,
)

EXECUTABLE_BITS = S_IXUSR | S_IXGRP | S_IXOTH
GIT_EXECUTABLE = "git"
GIT_LS_FILES_ARGS = ("ls-files", "--cached", "--")


def _candidate_roots(start: Path) -> list[Path]:
    start_path = start.resolve()
    if start_path.is_file():
        start_path = start_path.parent
    return list(chain([start_path], start_path.parents))


def is_repo_root(path: Path) -> bool:
    return (path / ".git").exists()


def is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return bool(path.stat().st_mode & EXECUTABLE_BITS)
    except OSError:
        return False


def tail_lines(text: str, max_lines: int = DEFAULT_SHORT_TAIL_LINES) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def parse_int_csv(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def truncate_float(value: float, decimals: int = DEFAULT_FLOAT_DECIMALS) -> float:
    factor = float(10**decimals)
    return int(value * factor) / factor


def fmt_float(value: float, decimals: int = DEFAULT_FLOAT_DECIMALS) -> str:
    return f"{truncate_float(value, decimals=decimals):.{decimals}f}"


def short_ref(value: str | None) -> str:
    """Truncate a git ref to 12 characters for display."""
    if not value:
        return "unknown"
    return value[:12]


def find_repo_root(start: Path | None = None) -> Path:
    env_root = environ.get(REPO_ROOT_ENV)
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if is_repo_root(env_path):
            return env_path

    search_from = (start or Path.cwd()).resolve()
    for candidate in _candidate_roots(search_from):
        if is_repo_root(candidate):
            return candidate

    raise RuntimeError(f"unable to discover repository root from: {search_from}")


def _sorted_relative(base_dir: Path, files: list[Path]) -> list[Path]:
    return sorted(files, key=lambda path: path.relative_to(base_dir).as_posix())


def list_files_recursive(base_dir: Path) -> list[Path]:
    return _sorted_relative(base_dir, [path for path in base_dir.rglob("*") if path.is_file()])


def list_tracked_files(base_dir: Path) -> list[Path] | None:
    try:
        repo_root = find_repo_root(base_dir)
        rel_base = base_dir.resolve().relative_to(repo_root)
    except (RuntimeError, ValueError):
        return None

    git_bin = shutil.which(GIT_EXECUTABLE)
    if git_bin is None:
        return None

    from .process import run as run_proc

    command = [git_bin, "-C", str(repo_root), *GIT_LS_FILES_ARGS, rel_base.as_posix()]
    try:
        result = run_proc(command, cwd=repo_root, check=True)
    except Exception:
        return None

    files: list[Path] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        candidate = (repo_root / line).resolve()
        if not candidate.is_file():
            continue
        try:
            candidate.relative_to(base_dir)
        except ValueError:
            continue
        files.append(candidate)

    return _sorted_relative(base_dir, files)


def list_repo_files_or_rglob(base_dir: Path) -> list[Path]:
    tracked_files = list_tracked_files(base_dir)
    if tracked_files is not None:
        return tracked_files
    return list_files_recursive(base_dir)


def read_utf8_lf(path: Path) -> str:
    """Read text using UTF-8 while normalizing CRLF to LF."""
    with path.open("rb") as file_obj:
        raw = file_obj.read()
    return raw.replace(b"\r\n", b"\n").decode(DEFAULT_UTF8_ENCODING)


def write_utf8_lf(path: Path, content: str) -> None:
    """Write text as UTF-8 with LF endings."""
    with path.open("wb") as file_obj:
        file_obj.write(content.encode(DEFAULT_UTF8_ENCODING))
