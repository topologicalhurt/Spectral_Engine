"""Hottest functions and lines (DETERMINISM_SURFACE_PLAN P4).

Per-PC dynamic instruction histogram (spectral_pchist.c TCG plugin) over the real
ARM binary under qemu, mapped back to inlined function + source line via
addr2line on a -g ELF. Promotes the by-hand PC-histogram profiling to a
first-class command. Counts are MEASURED (qemu-tcg retired instructions) — not
cycles; for cycles use the llvm-mca census (m7-census).
"""

from __future__ import annotations

import platform
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ...core.process import run
from .counts import RESULT_RE, CountsError, _build_runner_elf
from .fixture import WorkloadFixture, default_fixture
from .toolchain import NATIVE_DIR, Toolchain

_HIST_RE = re.compile(r"^([0-9a-f]+)\s+(\d+)$", re.M)
_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]+$")
_A2L_CHUNK = 256


def _build_pchist_plugin(tc: Toolchain, out_dir: Path) -> Path:
    src = NATIVE_DIR / "qemu" / "spectral_pchist.c"
    plugin = out_dir / "libspectral_pchist.so"
    ldflags = ["-Wl,-undefined,dynamic_lookup"] if platform.system() == "Darwin" else []
    result = run(
        ["cc", "-O2", "-shared", "-fPIC", *tc.glib_cflags,
         "-I", str(tc.qemu_include_dir), *ldflags,
         str(src), "-o", str(plugin), *tc.glib_libs],
        cwd=tc.repo_root, check=False)
    if result.returncode != 0:
        raise CountsError(f"pchist plugin build failed:\n{result.stderr}")
    return plugin


def _run_pchist(tc: Toolchain, elf: Path, plugin: Path, hist_path: Path) -> str:
    result = run(
        [str(tc.qemu_system_arm), "-M", "mps2-an500", "-semihosting",
         "-display", "none", "-serial", "none", "-monitor", "none",
         "-plugin", f"{plugin},out={hist_path}", "-kernel", str(elf)],
        cwd=tc.repo_root, timeout_sec=300, check=False)
    out = (result.stdout + "\n" + result.stderr).replace("\r", "")
    verdict = RESULT_RE.search(out)
    if result.returncode != 0 or verdict is None or verdict.group(1) != "PASS":
        raise CountsError(
            f"pchist workload failed (rc={result.returncode}, "
            f"result={verdict.group(1) if verdict else 'missing'}):\n{out}")
    return hist_path.read_text(encoding="utf-8")


def _addr2line(tc: Toolchain, elf: Path, pcs: list[int]) -> dict[int, tuple[str, str]]:
    """PC -> (innermost function, file:line) via addr2line -afiC, chunked."""
    a2l = Path(tc.arm_gcc).parent / "arm-none-eabi-addr2line"
    out: dict[int, tuple[str, str]] = {}
    for i in range(0, len(pcs), _A2L_CHUNK):
        chunk = pcs[i:i + _A2L_CHUNK]
        result = run([str(a2l), "-afiC", "-e", str(elf),
                      *[f"0x{p:x}" for p in chunk]], cwd=tc.repo_root, check=False)
        cur_pc: int | None = None
        func: str | None = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if _ADDR_RE.match(line):
                cur_pc = int(line, 16)
                func = None
            elif cur_pc is not None and func is None:
                func = line                              # innermost frame fn
            elif cur_pc is not None and func is not None:
                if cur_pc not in out:                    # keep the INNERMOST frame only
                    out[cur_pc] = (func, line)           # (function, file:line)
    return out


def analyze(tc: Toolchain, *, out_dir: Path,
            fixture: WorkloadFixture | None = None, top: int = 20) -> dict[str, Any]:
    if tc.qemu_system_arm is None:
        raise CountsError("qemu-system-arm unavailable")
    spec = fixture or default_fixture()
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = _build_runner_elf(tc, spec, out_dir, extra_cflags=("-g",))
    plugin = _build_pchist_plugin(tc, out_dir)
    hist = _run_pchist(tc, elf, plugin, out_dir / "pchist.txt")

    pc_counts = {int(m.group(1), 16): int(m.group(2)) for m in _HIST_RE.finditer(hist)}
    if not pc_counts:
        raise CountsError("empty PC histogram")
    total = sum(pc_counts.values())
    mapping = _addr2line(tc, elf, sorted(pc_counts))

    by_func: dict[str, int] = defaultdict(int)
    by_line: dict[str, int] = defaultdict(int)
    for pc, n in pc_counts.items():
        func, fileline = mapping.get(pc, ("??", "??"))
        by_func[func] += n
        by_line[f"{fileline}  [{func}]"] += n

    def rank(d: dict[str, int]) -> list[dict[str, Any]]:
        return [{"name": k, "insns": v, "pct": round(100.0 * v / total, 2)}
                for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:top]]

    return {
        "provenance": "measured: qemu-tcg per-PC retired instructions, mapped via "
                      "addr2line -i on a -g ELF (inlined function + line); not cycles",
        "fixture": {"name": spec.name, "digest": spec.digest()},
        "total_insns": total,
        "unique_pcs": len(pc_counts),
        "hottest_functions": rank(by_func),
        "hottest_lines": rank(by_line),
    }
