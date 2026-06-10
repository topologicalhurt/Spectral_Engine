"""Layer 1: exact dynamic counts of the real kernel under QEMU [measured].

Builds the freestanding mps2-an500 runner from the production TUs + the
fixture header (fixture SSOT), builds the TCG counting plugin against the
installed qemu's plugin API, runs the workload, and parses the per-symbol
counts table. Counts are retired instructions and load/store bytes — never
cycles (TCG has no timing model).

Reproducibility is part of the contract: by default the workload runs twice
and the counts must be bit-identical, otherwise the run fails loudly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.process import run
from .fixture import HEADER_NAME, WorkloadFixture, default_fixture
from .toolchain import NATIVE_DIR, Toolchain

RANGE_SYMBOLS = (
    "spectral_arm32_process",
    "spectral_arm32_load",
    "spectral_lut_init_sine",
    "memcpy",
    "memset",
    "main",
)

NM_LINE_RE = re.compile(r"^([0-9a-f]+)\s+([0-9a-f]+)\s+[Tt]\s+(\S+)$", re.MULTILINE)
COUNTS_ROW_RE = re.compile(
    r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.MULTILINE
)
REGION_LINE_RE = re.compile(
    r"^# data bytes by region: code=(\d+) ssram23=(\d+) bulk60=(\d+) other=(\d+)",
    re.MULTILINE,
)
RESULT_RE = re.compile(r"^RESULT: (PASS|FAIL.*)$", re.MULTILINE)
CHECKSUM_RE = re.compile(r"^CHECKSUM=([0-9a-f]{8})$", re.MULTILINE)
RENDERED_RE = re.compile(r"^RENDERED=([0-9a-f]{8})$", re.MULTILINE)


class CountsError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RangeCounts:
    name: str
    insns: int
    loads: int
    stores: int
    load_bytes: int
    store_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "insns": self.insns,
            "loads": self.loads,
            "stores": self.stores,
            "load_bytes": self.load_bytes,
            "store_bytes": self.store_bytes,
        }


@dataclass(frozen=True, slots=True)
class CountsReport:
    """[measured: qemu-tcg dynamic counts] — counts, never cycles."""
    fixture_name: str
    fixture_digest: str
    checksum: str
    rendered_samples: int
    ranges: tuple[RangeCounts, ...]
    region_bytes: dict[str, int]
    reproducible: bool

    def range(self, name: str) -> RangeCounts | None:
        for item in self.ranges:
            if item.name == name:
                return item
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": "measured: qemu-tcg dynamic counts (not cycles)",
            "fixture": {"name": self.fixture_name, "digest": self.fixture_digest},
            "checksum": self.checksum,
            "rendered_samples": self.rendered_samples,
            "reproducible": self.reproducible,
            "ranges": [item.as_dict() for item in self.ranges],
            "region_bytes": dict(self.region_bytes),
        }


def _build_runner_elf(tc: Toolchain, fixture: WorkloadFixture, out_dir: Path) -> Path:
    qemu_dir = NATIVE_DIR / "qemu"
    arm_backend_dir = tc.repo_root / "spectral_engine/synth/backends/arm"
    (out_dir / HEADER_NAME).write_text(fixture.to_c_header(), encoding="utf-8")

    gc_flags = ("-ffunction-sections", "-fdata-sections")
    objects: list[Path] = []
    compile_plan: list[tuple[Path, tuple[str, ...]]] = [
        # startup: -fno-builtin + no loop-distribute so GCC cannot rewrite the
        # mem loops into calls to themselves.
        (qemu_dir / "startup.c", ("-fno-builtin", "-fno-tree-loop-distribute-patterns")),
        (qemu_dir / "qemu_main.c", ("-fno-builtin",)),
        *((tu, ()) for tu in tc.kernel_tus()),
    ]
    for src, extra in compile_plan:
        obj = out_dir / (src.stem + ".o")
        result = run(
            [tc.arm_gcc, *tc.cflags(extra_includes=(arm_backend_dir, out_dir)),
             *gc_flags, *extra, "-c", str(src), "-o", str(obj)],
            cwd=tc.repo_root,
            check=False,
        )
        if result.returncode != 0:
            raise CountsError(f"compile failed for {src.name}:\n{result.stderr}")
        objects.append(obj)

    elf = out_dir / "qemu_counts.elf"
    result = run(
        [tc.arm_gcc, "-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard",
         "-nostdlib", "-T", str(qemu_dir / "mps2_an500.ld"), "-Wl,--gc-sections",
         *map(str, objects), "-lgcc", "-o", str(elf)],
        cwd=tc.repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise CountsError(f"runner link failed:\n{result.stderr}")
    return elf


def _symbol_ranges(tc: Toolchain, elf: Path) -> list[tuple[str, int, int]]:
    if tc.arm_nm is None:
        raise CountsError("arm-none-eabi-nm not found; cannot derive symbol ranges")
    result = run([tc.arm_nm, "-S", str(elf)], cwd=tc.repo_root, check=False)
    if result.returncode != 0:
        raise CountsError(f"nm failed:\n{result.stderr}")

    ranges: list[tuple[str, int, int]] = []
    for match in NM_LINE_RE.finditer(result.stdout):
        addr, size, name = int(match.group(1), 16), int(match.group(2), 16), match.group(3)
        if name in RANGE_SYMBOLS:
            ranges.append((name, addr, addr + size))
    missing = sorted(set(RANGE_SYMBOLS) - {name for name, _, _ in ranges})
    if missing:
        raise CountsError(f"symbols missing from runner ELF (gc-sections drift?): {missing}")
    return sorted(ranges)


def _build_plugin(tc: Toolchain, out_dir: Path) -> Path:
    plugin_src = NATIVE_DIR / "qemu" / "spectral_counts.c"
    plugin = out_dir / "libspectral_counts.so"
    import platform

    ldflags = ["-Wl,-undefined,dynamic_lookup"] if platform.system() == "Darwin" else []
    result = run(
        ["cc", "-O2", "-shared", "-fPIC", *tc.glib_cflags,
         "-I", str(tc.qemu_include_dir), *ldflags,
         str(plugin_src), "-o", str(plugin), *tc.glib_libs],
        cwd=tc.repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise CountsError(f"plugin build failed:\n{result.stderr}")
    return plugin


def _run_once(
    tc: Toolchain, elf: Path, plugin: Path,
    ranges: list[tuple[str, int, int]], counts_path: Path,
) -> tuple[str, str, int]:
    range_args = "".join(
        f",range={name}:0x{start:x}:0x{end:x}" for name, start, end in ranges
    )
    result = run(
        [str(tc.qemu_system_arm), "-M", "mps2-an500", "-semihosting",
         "-display", "none", "-serial", "none", "-monitor", "none",
         "-plugin", f"{plugin}{range_args},out={counts_path}",
         "-kernel", str(elf)],
        cwd=tc.repo_root,
        timeout_sec=300,
        check=False,
    )
    # Semihosting console lands on stderr in this qemu config; accept either
    # stream and normalize the char backend's \r\n.
    stdout = (result.stdout + "\n" + result.stderr).replace("\r", "")
    verdict = RESULT_RE.search(stdout)
    if result.returncode != 0 or verdict is None or verdict.group(1) != "PASS":
        raise CountsError(
            f"qemu workload failed (rc={result.returncode}, "
            f"result={verdict.group(1) if verdict else 'missing'}):\n{stdout}\n{result.stderr}"
        )
    checksum = CHECKSUM_RE.search(stdout)
    rendered = RENDERED_RE.search(stdout)
    if checksum is None or rendered is None:
        raise CountsError(f"workload output missing CHECKSUM/RENDERED lines:\n{stdout}")
    return counts_path.read_text(encoding="utf-8"), checksum.group(1), int(rendered.group(1), 16)


def _parse_counts(table: str) -> tuple[tuple[RangeCounts, ...], dict[str, int]]:
    ranges = tuple(
        RangeCounts(
            name=m.group(1), insns=int(m.group(2)), loads=int(m.group(3)),
            stores=int(m.group(4)), load_bytes=int(m.group(5)), store_bytes=int(m.group(6)),
        )
        for m in COUNTS_ROW_RE.finditer(table)
    )
    if not ranges:
        raise CountsError(f"no count rows parsed from plugin output:\n{table}")
    region = REGION_LINE_RE.search(table)
    if region is None:
        raise CountsError("region-bytes line missing from plugin output")
    region_bytes = {
        "code": int(region.group(1)),
        "ssram23": int(region.group(2)),
        "bulk60": int(region.group(3)),
        "other": int(region.group(4)),
    }
    return ranges, region_bytes


def measure(
    tc: Toolchain,
    *,
    out_dir: Path,
    fixture: WorkloadFixture | None = None,
    verify_reproducible: bool = True,
) -> CountsReport:
    """Build + run the counts rig; returns measured counts for the fixture."""
    if tc.qemu_system_arm is None:
        raise CountsError("qemu-system-arm unavailable; run toolchain.discover(need={'qemu'})")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = fixture or default_fixture()
    spec.validate()

    elf = _build_runner_elf(tc, spec, out_dir)
    ranges = _symbol_ranges(tc, elf)
    plugin = _build_plugin(tc, out_dir)

    table_1, checksum, rendered = _run_once(tc, elf, plugin, ranges, out_dir / "qemu_counts.txt")
    reproducible = True
    if verify_reproducible:
        table_2, checksum_2, _ = _run_once(tc, elf, plugin, ranges, out_dir / "qemu_counts_verify.txt")
        reproducible = (table_1 == table_2) and (checksum == checksum_2)
        if not reproducible:
            raise CountsError(
                "counts differ between identical runs — the rig is broken "
                "(nondeterminism in the runner, plugin, or machine model)"
            )

    parsed_ranges, region_bytes = _parse_counts(table_1)
    if rendered != spec.total_samples:
        raise CountsError(
            f"workload rendered {rendered} samples, fixture expects {spec.total_samples}"
        )

    return CountsReport(
        fixture_name=spec.name,
        fixture_digest=spec.digest(),
        checksum=checksum,
        rendered_samples=rendered,
        ranges=parsed_ranges,
        region_bytes=region_bytes,
        reproducible=reproducible,
    )
