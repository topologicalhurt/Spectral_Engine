"""Console formatting helpers shared across tool scripts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from os import environ

from .constants import NO_COLOR_ENV


@dataclass(slots=True)
class Console:
    use_color: bool = True

    def __post_init__(self) -> None:
        self.use_color = bool(self.use_color and sys.stdout.isatty() and environ.get(NO_COLOR_ENV) is None)

    def green(self, text: str) -> str:
        return f"\033[32m{text}\033[0m" if self.use_color else text

    def red(self, text: str) -> str:
        return f"\033[31m{text}\033[0m" if self.use_color else text

    def yellow(self, text: str) -> str:
        return f"\033[33m{text}\033[0m" if self.use_color else text

    def cyan(self, text: str) -> str:
        return f"\033[36m{text}\033[0m" if self.use_color else text

    def bold(self, text: str) -> str:
        return f"\033[1m{text}\033[0m" if self.use_color else text

    def status(self, value: str) -> str:
        norm = value.lower()
        if norm == "ok":
            return self.green("ok")
        if norm == "failed":
            return self.red("FAILED")
        if norm == "skipped":
            return self.yellow("skipped")
        return value
