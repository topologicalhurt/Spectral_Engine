from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

JSON_INDENT = 2
JSON_NEWLINE = "\n"


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def serialize_report(report: dict[str, Any]) -> str:
    safe_report = _sanitize_json_value(report)
    return json.dumps(safe_report, indent=JSON_INDENT, sort_keys=True, allow_nan=False)


def render_report(report: dict[str, Any], *, raw: bool, use_color: bool = True) -> str:
    if raw:
        return serialize_report(report)
    from .branch_view import BranchFormatter
    return BranchFormatter(use_color=use_color).render(report)


def write_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_report(report) + JSON_NEWLINE, encoding="utf-8")
