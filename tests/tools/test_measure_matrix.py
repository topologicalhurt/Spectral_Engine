"""Per-target binary resolution for the measurement matrix.

The desktop and simulate outputs coexist in build/bin, so a shared or wrong
glob makes a matrix row silently measure the WRONG build — the bug class
this file pins. The chain asserted here is fully derived from the CMake
SSOTs (C-truth rule), end to end:

  profile.build_target -> set_target_properties(... OUTPUT_NAME ${VAR})
  -> set(VAR "spectral_...") name expression -> derived glob
  -> matches its own target's concrete name and NO other target's.
"""

from __future__ import annotations

import re
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from spectral_tools.performance import matrix  # noqa: E402
from spectral_tools.testing.benchmark_workflow import Performance  # noqa: E402

HOST_PROFILES = [p for p in matrix.TARGETS if p.binary_cmake_var is not None]

_OUTPUT_NAME_RE = re.compile(
    r"set_target_properties\((\w+)\s+PROPERTIES\s+OUTPUT_NAME\s+\$\{(TARGET_[A-Z_]+)\}\)"
)


def _output_name_bindings() -> dict[str, str]:
    """cmake build target -> TARGET_* naming variable, parsed from the
    target definitions (the binding SSOT)."""
    bindings: dict[str, str] = {}
    for path in (ROOT / "spectral_engine" / "cmake" / "targets").glob("*.cmake"):
        for build_target, var in _OUTPUT_NAME_RE.findall(
                path.read_text(encoding="utf-8")):
            bindings[build_target] = var
    return bindings


def _concrete(name_expr: str) -> str:
    """Expand ${...} the way a configured build would: any single token."""
    return re.sub(r"\$\{[^}]+\}", "x", name_expr)


def test_matrix_has_host_run_profiles():
    assert {p.name for p in HOST_PROFILES} >= {"desktop", "simulate"}


def test_profile_binding_matches_cmake_output_name():
    """A profile claiming TARGET_X must be the profile whose cmake build
    target actually emits that OUTPUT_NAME (a swapped binding here was the
    original silent-wrong-binary bug class)."""
    bindings = _output_name_bindings()
    assert bindings, "OUTPUT_NAME bindings not found — the binding SSOT moved"
    for profile in HOST_PROFILES:
        assert bindings.get(profile.build_target) == profile.binary_cmake_var, (
            f"profile '{profile.name}' (build target '{profile.build_target}') "
            f"claims {profile.binary_cmake_var}, but cmake binds "
            f"{bindings.get(profile.build_target)}")


def test_every_host_glob_matches_own_name_and_no_other():
    names = matrix.cmake_target_names(ROOT)
    for profile in HOST_PROFILES:
        glob = matrix.binary_glob(profile, ROOT)
        own = _concrete(names[profile.binary_cmake_var])
        assert fnmatch(own, glob), f"{profile.name}: {glob} misses own {own}"
        for var, expr in names.items():
            if var == profile.binary_cmake_var:
                continue
            assert not fnmatch(_concrete(expr), glob), (
                f"profile '{profile.name}' glob '{glob}' also matches "
                f"{var}='{expr}' — resolution is ambiguous")


def test_non_host_profiles_fail_loudly():
    with pytest.raises(KeyError):
        matrix.binary_glob(matrix.target("m7"), ROOT)


def test_resolve_target_binary_is_per_target(tmp_path):
    """End-to-end through the workflow layer: with both binaries present,
    each target resolves its own, never the other's."""
    names = matrix.cmake_target_names(ROOT)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    made = {}
    for profile in HOST_PROFILES:
        path = bin_dir / _concrete(names[profile.binary_cmake_var])
        path.write_bytes(b"#!/bin/sh\n")
        path.chmod(0o755)
        made[profile.name] = path
    perf = Performance(repo_root=ROOT)
    for profile in HOST_PROFILES:
        resolved = perf.resolve_target_binary(tmp_path, profile.name)
        assert resolved == made[profile.name], (
            f"'{profile.name}' resolved {resolved.name}, "
            f"expected {made[profile.name].name}")
