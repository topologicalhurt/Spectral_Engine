"""Layer 0/2: codegen census [measured] + llvm-mca loop analysis [modeled].

Census compiles the REAL production TU (spectral_synth_arm32.c) with the Daisy
flags and counts the DSP/MAC instructions actually emitted — a measured,
assumption-free quantity. Loop analysis compiles the marker wrapper TU
(native/kernel_wrappers.c), extracts each kernel's innermost loop bodies, and
runs the Arm-contributed CortexM7Model in llvm-mca over each — modeled,
perfect-memory steady-state cycles. The two layers are never mixed in one
number; results carry their provenance tag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.process import run
from .toolchain import NATIVE_DIR, Toolchain

# Instructions worth tracking in the census: DSP/MAC/saturation ops the S1
# work targets, FPU double ops (oscillator seeding), and prefetches.
CENSUS_MNEMONICS = (
    "smulbb", "smlald", "smlad", "smlabb", "qadd16", "qsub16", "ssat",
    "qadd", "qsub", "smull", "smlal", "umull", "umlal", "mla", "mls",
    "vldr", "vstr", "vmul.f64", "vadd.f64", "pld",
)
_CENSUS_RE = re.compile(
    r"^\s+(" + "|".join(re.escape(m) for m in CENSUS_MNEMONICS) + r")\b"
)

MCA_BEGIN_RE = re.compile(r"#\s*LLVM-MCA-BEGIN\s+(\S+)")
MCA_END_RE = re.compile(r"#\s*LLVM-MCA-END")
LABEL_RE = re.compile(r"^(\.?L[\w.]+):")
# Conditional/unconditional branch to a local label; excludes bl/blx/bx
# (calls, returns) and cbz/cbnz (forward-only by ISA definition).
BRANCH_RE = re.compile(
    r"^\s+(b(?:ne|eq|cs|cc|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)?(?:\.[wn])?)\s+(\.?L[\w.]+)\b"
)
DIRECTIVE_RE = re.compile(r"^\s*\.")

MCA_REGION_RE = re.compile(r"^\[(\d+)\] Code Region - (\S+)", re.MULTILINE)
MCA_FIELD_RES = {
    "iterations": re.compile(r"^Iterations:\s+(\d+)", re.MULTILINE),
    "instructions": re.compile(r"^Instructions:\s+(\d+)", re.MULTILINE),
    "total_cycles": re.compile(r"^Total Cycles:\s+(\d+)", re.MULTILINE),
    "total_uops": re.compile(r"^Total uOps:\s+(\d+)", re.MULTILINE),
}


class CodegenError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CensusResult:
    """[measured: codegen] instruction census of the production M7 TU."""
    tu: str
    compiler: str
    counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": "measured: codegen",
            "tu": self.tu,
            "compiler": self.compiler,
            "counts": dict(sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        }


@dataclass(frozen=True, slots=True)
class LoopBody:
    kernel: str
    label: str
    instructions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoopAnalysis:
    """[modeled: llvm-mca/CortexM7Model, perfect memory] one innermost loop."""
    kernel: str
    label: str
    instructions_per_iter: int
    cycles_per_iter: float
    uops_per_iter: float
    ipc: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": "modeled: llvm-mca/CortexM7Model (perfect memory)",
            "kernel": self.kernel,
            "label": self.label,
            "instructions_per_iter": self.instructions_per_iter,
            "cycles_per_iter": self.cycles_per_iter,
            "uops_per_iter": self.uops_per_iter,
            "ipc": self.ipc,
        }


@dataclass(frozen=True, slots=True)
class CodegenReport:
    census: CensusResult
    loops: tuple[LoopAnalysis, ...]
    failed_regions: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "census": self.census.as_dict(),
            "loops": [loop.as_dict() for loop in self.loops],
            "failed_regions": list(self.failed_regions),
        }


def _marker_regions(lines: list[str]) -> list[tuple[str, list[str]]]:
    regions: list[tuple[str, list[str]]] = []
    name: str | None = None
    buf: list[str] = []
    for line in lines:
        begin = MCA_BEGIN_RE.search(line)
        if begin:
            name, buf = begin.group(1), []
            continue
        if MCA_END_RE.search(line):
            if name is not None:
                regions.append((name, buf))
            name = None
            continue
        if name is not None:
            buf.append(line)
    return regions


def _innermost_loops(body: list[str]) -> list[tuple[int, int, str]]:
    label_at: dict[str, int] = {}
    spans: list[tuple[int, int, str]] = []
    for i, line in enumerate(body):
        label = LABEL_RE.match(line)
        if label:
            label_at[label.group(1)] = i
            continue
        branch = BRANCH_RE.match(line)
        if branch and branch.group(2) in label_at:
            spans.append((label_at[branch.group(2)], i, branch.group(2)))
    # Collapse multiple back-edges to the same loop-top into one span reaching
    # the FARTHEST back-edge. A `continue`/early-iterate adds an earlier
    # back-edge to the same label; keeping the smaller span would slice a
    # truncated loop body and silently under-report modeled cycles/iter.
    by_start: dict[int, tuple[int, int, str]] = {}
    for span in spans:
        current = by_start.get(span[0])
        if current is None or span[1] > current[1]:
            by_start[span[0]] = span
    collapsed = list(by_start.values())
    return [
        s for s in collapsed
        if not any(o is not s and s[0] <= o[0] and o[1] <= s[1] for o in collapsed)
    ]


def extract_loop_bodies(asm_text: str) -> list[LoopBody]:
    """Innermost loop bodies (branch included — it costs an issue slot) from
    every marker-bracketed kernel region in GCC assembly output."""
    loops: list[LoopBody] = []
    for kernel, body in _marker_regions(asm_text.splitlines()):
        for start, end, label in _innermost_loops(body):
            insns = tuple(
                line for line in body[start + 1 : end + 1]
                if line.strip() and not DIRECTIVE_RE.match(line) and not LABEL_RE.match(line)
            )
            if insns:
                loops.append(LoopBody(kernel=kernel, label=label, instructions=insns))
    return loops


def census(tc: Toolchain, *, out_dir: Path) -> CensusResult:
    """Compile the production ARM TU and count emitted DSP/MAC instructions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tu = tc.repo_root / "spectral_engine/synth/backends/arm/spectral_synth_arm32.c"
    asm_path = out_dir / "arm32_m7.s"

    result = run(
        [tc.arm_gcc, *tc.cflags(), "-S", str(tu), "-o", str(asm_path)],
        cwd=tc.repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise CodegenError(f"census compile failed:\n{result.stderr}")

    version = run([tc.arm_gcc, "--version"], cwd=tc.repo_root, check=False)
    compiler = version.stdout.splitlines()[0] if version.stdout else tc.arm_gcc

    counts: dict[str, int] = {}
    for line in asm_path.read_text(encoding="utf-8").splitlines():
        match = _CENSUS_RE.match(line)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1

    return CensusResult(tu=str(tu.relative_to(tc.repo_root)), compiler=compiler, counts=counts)


def _parse_mca_report(report: str, region_names: list[str]) -> list[LoopAnalysis]:
    """Parse per-region summaries out of an llvm-mca multi-region report."""
    analyses: list[LoopAnalysis] = []
    matches = list(MCA_REGION_RE.finditer(report))
    for idx, match in enumerate(matches):
        seg_start = match.end()
        seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        segment = report[seg_start:seg_end]

        fields: dict[str, int] = {}
        for key, regex in MCA_FIELD_RES.items():
            value = regex.search(segment)
            if value:
                fields[key] = int(value.group(1))
        if not {"iterations", "instructions", "total_cycles"} <= fields.keys():
            continue

        iters = fields["iterations"]
        name = match.group(2)
        kernel, _, label = name.partition("/")
        analyses.append(
            LoopAnalysis(
                kernel=kernel,
                label=label or name,
                instructions_per_iter=fields["instructions"] // iters,
                cycles_per_iter=fields["total_cycles"] / iters,
                uops_per_iter=fields.get("total_uops", fields["instructions"]) / iters,
                ipc=fields["instructions"] / fields["total_cycles"],
            )
        )
    return analyses


def loop_analysis(
    tc: Toolchain,
    *,
    out_dir: Path,
    iterations: int = 100,
    expected_kernels: tuple[str, ...] = ("synth_core_m7", "synth_core_pair_m7", "synth_fade_m7"),
) -> tuple[tuple[LoopAnalysis, ...], tuple[dict[str, str], ...]]:
    """Extract innermost loops from the wrapper TU and run llvm-mca per loop.

    Each region is analyzed in its own mca invocation so one unsupported
    instruction cannot take down the whole report; failures are returned, not
    swallowed. Raises if an expected kernel yields no loops at all (marker or
    inlining drift — the rig would otherwise silently measure nothing).
    """
    if tc.llvm_mca is None:
        raise CodegenError("llvm-mca unavailable; run toolchain.discover(need={'mca'})")
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = NATIVE_DIR / "kernel_wrappers.c"
    asm_path = out_dir / "kernel_wrappers.s"
    arm_backend_dir = tc.repo_root / "spectral_engine/synth/backends/arm"

    result = run(
        [tc.arm_gcc, *tc.cflags(extra_includes=(arm_backend_dir,)),
         "-S", str(wrapper), "-o", str(asm_path)],
        cwd=tc.repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise CodegenError(f"wrapper compile failed:\n{result.stderr}")

    loops = extract_loop_bodies(asm_path.read_text(encoding="utf-8"))
    seen_kernels = {loop.kernel for loop in loops}
    missing = [k for k in expected_kernels if k not in seen_kernels]
    if missing:
        raise CodegenError(
            f"no innermost loops extracted for kernel(s) {missing}: "
            "marker wrappers or branch-pattern extraction have drifted from codegen"
        )

    analyses: list[LoopAnalysis] = []
    failures: list[dict[str, str]] = []
    for loop in loops:
        name = f"{loop.kernel}/{loop.label}"
        try:
            analyses.extend(
                mca_region(
                    tc, name=name, instructions=loop.instructions,
                    out_dir=out_dir, file_stem=f"loop_{loop.kernel}_{loop.label.strip('.')}",
                    iterations=iterations,
                )
            )
        except CodegenError as exc:
            failures.append({"region": name, "error": str(exc)})

    return tuple(analyses), tuple(failures)


def mca_region(
    tc: Toolchain,
    *,
    name: str,
    instructions: tuple[str, ...],
    out_dir: Path,
    file_stem: str,
    iterations: int = 100,
) -> list[LoopAnalysis]:
    """Run one marker-bracketed region through llvm-mca/CortexM7Model.

    The single mca invocation path shared by the kernel loop analysis and the
    validation microbench set, so both measure under identical flags."""
    if tc.llvm_mca is None:
        raise CodegenError("llvm-mca unavailable; run toolchain.discover(need={'mca'})")
    out_dir.mkdir(parents=True, exist_ok=True)
    region_src = "\n".join(
        ["\t.syntax unified", "\t.thumb", f"# LLVM-MCA-BEGIN {name}",
         *instructions, "# LLVM-MCA-END", ""]
    )
    region_path = out_dir / f"{file_stem}.s"
    region_path.write_text(region_src, encoding="utf-8")

    mca = run(
        [tc.llvm_mca, "-mtriple=thumbv7em-none-eabi", "-mcpu=cortex-m7",
         f"-iterations={iterations}", str(region_path)],
        cwd=tc.repo_root,
        check=False,
    )
    if mca.returncode != 0:
        raise CodegenError(
            mca.stderr.strip().splitlines()[-1] if mca.stderr else "mca failed"
        )
    return _parse_mca_report(mca.stdout, [name])


def codegen_report(tc: Toolchain, *, out_dir: Path) -> CodegenReport:
    cen = census(tc, out_dir=out_dir)
    loops, failures = loop_analysis(tc, out_dir=out_dir)
    return CodegenReport(census=cen, loops=loops, failed_regions=failures)
