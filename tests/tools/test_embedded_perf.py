"""Harness tests for spectral_tools.performance.embedded (M7 measurement stack).

Unit tests cover the parsing/extraction logic with canned inputs (no
toolchain needed). Integration tests exercise the real rigs end-to-end and
skip cleanly when the cross toolchain / qemu are absent.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from spectral_tools.performance.embedded import codegen, counts, fixture, toolchain  # noqa: E402


# --- fixture SSOT -----------------------------------------------------------

def test_fixture_header_is_deterministic():
    spec = fixture.default_fixture()
    assert spec.to_c_header() == spec.to_c_header()
    assert spec.digest() == spec.digest()


def test_fixture_header_contains_every_voice_and_macros():
    spec = fixture.default_fixture()
    header = spec.to_c_header()
    assert f"#define FIXTURE_N_SEG   {len(spec.voices)}u" in header
    assert f"#define FIXTURE_TOTAL   {spec.total_samples}u" in header
    assert header.count("X(") == len(spec.voices)
    assert spec.digest() in header


def test_fixture_validation_rejects_bad_specs():
    base = fixture.default_fixture()
    with pytest.raises(ValueError, match="monotonic"):
        fixture.WorkloadFixture(
            name="bad", sample_rate=48000, total_samples=512, block_samples=256,
            voices=(fixture.Voice(1024, 256, 220.0, 0.1), fixture.Voice(0, 256, 220.0, 0.1)),
        ).validate()
    with pytest.raises(ValueError, match="uint16"):
        fixture.Voice(0, 0x10000, 220.0, 0.1).validate()
    with pytest.raises(ValueError, match="multiple"):
        fixture.WorkloadFixture(
            name="bad", sample_rate=48000, total_samples=300, block_samples=256,
            voices=base.voices,
        ).validate()


def test_fixture_digest_tracks_content():
    base = fixture.default_fixture()
    changed = fixture.WorkloadFixture(
        name=base.name, sample_rate=base.sample_rate, total_samples=base.total_samples,
        block_samples=base.block_samples,
        voices=base.voices[:-1] + (fixture.Voice(8192, 8192, 1979.0, 0.05),),
    )
    assert changed.digest() != base.digest()


# --- target-flag SSOT pairing against options.cmake -------------------------

def test_m7_flags_match_daisy_cmake_options():
    """toolchain.M7_TARGET_FLAGS duplicates the Daisy production flags; this
    pairing test makes silent drift impossible (AI_CANON SSOT rule)."""
    options = (ROOT / "spectral_engine/cmake/options.cmake").read_text(encoding="utf-8")
    for var in ("SPECTRAL_DAISY_CPU", "SPECTRAL_DAISY_FPU", "SPECTRAL_DAISY_FLOAT_ABI"):
        match = re.search(rf'set\({var} "([^"]+)"', options)
        assert match, f"{var} missing from options.cmake"
        assert match.group(1) in toolchain.M7_TARGET_FLAGS, (
            f"{var}={match.group(1)} not in M7_TARGET_FLAGS — embedded measurements "
            "would run with different flags than the Daisy production build"
        )


# --- loop extraction --------------------------------------------------------

CANNED_ASM = """\
\t.text
# LLVM-MCA-BEGIN kernel_a
\tmov r0, #0
.L1:
\tadd r1, r1, #1
.L2:
\tsmlald r2, r3, r4, r5
\tbne .L2
\tcmp r1, r6
\tblt .L1
# LLVM-MCA-END
# LLVM-MCA-BEGIN kernel_b
.L9:
\tldr r0, [r1], #4
\tcbz r0, .L10
\tbgt .L9
.L10:
\tbx lr
# LLVM-MCA-END
"""


def test_extract_innermost_loops_only():
    loops = codegen.extract_loop_bodies(CANNED_ASM)
    by_kernel = {(l.kernel, l.label) for l in loops}
    # kernel_a: .L2 is innermost (nested inside .L1's span); .L1 must NOT appear.
    assert ("kernel_a", ".L2") in by_kernel
    assert ("kernel_a", ".L1") not in by_kernel
    # kernel_b: backward bgt .L9 forms a loop; cbz is forward-only and ignored.
    assert ("kernel_b", ".L9") in by_kernel
    inner = next(l for l in loops if l.label == ".L2")
    assert inner.instructions == ("\tsmlald r2, r3, r4, r5", "\tbne .L2")


def test_extract_ignores_labels_directives_and_calls():
    asm = """\
# LLVM-MCA-BEGIN k
.L5:
\t.align 2
\tbl helper
\tsubs r0, r0, #1
\tbne .L5
# LLVM-MCA-END
"""
    loops = codegen.extract_loop_bodies(asm)
    assert len(loops) == 1
    assert loops[0].instructions == ("\tbl helper", "\tsubs r0, r0, #1", "\tbne .L5")


# --- mca report parsing -----------------------------------------------------

CANNED_MCA = """\
[0] Code Region - synth_core_m7/.L452

Iterations:        100
Instructions:      43000
Total Cycles:      34002
Total uOps:        43000

Dispatch Width:    2
IPC:               1.26
"""


def test_parse_mca_report():
    analyses = codegen._parse_mca_report(CANNED_MCA, ["synth_core_m7/.L452"])
    assert len(analyses) == 1
    loop = analyses[0]
    assert loop.kernel == "synth_core_m7"
    assert loop.label == ".L452"
    assert loop.instructions_per_iter == 430
    assert loop.cycles_per_iter == pytest.approx(340.02)
    assert loop.ipc == pytest.approx(43000 / 34002)


# --- counts table parsing ---------------------------------------------------

CANNED_COUNTS = """\
# [measured: qemu-tcg dynamic counts] — not cycles
range                                 insns        loads       stores     load_bytes    store_bytes
<other>                              498145        74155        24258         460452          94394
spectral_arm32_process              3763544      1097748       353301        4566738        1652298
# data bytes by region: code=508272 ssram23=6324410 bulk60=0 other=8
"""


def test_parse_counts_table():
    ranges, regions = counts._parse_counts(CANNED_COUNTS)
    names = {r.name for r in ranges}
    assert names == {"<other>", "spectral_arm32_process"}
    process = next(r for r in ranges if r.name == "spectral_arm32_process")
    assert process.insns == 3763544
    assert process.load_bytes == 4566738
    assert regions == {"code": 508272, "ssram23": 6324410, "bulk60": 0, "other": 8}


def test_parse_counts_rejects_garbage():
    with pytest.raises(counts.CountsError, match="no count rows"):
        counts._parse_counts("nothing useful here\n")


# --- integration (skipped when toolchain absent) -----------------------------

HAVE_ARM_GCC = shutil.which("arm-none-eabi-gcc") is not None
HAVE_MCA = toolchain._find_llvm_mca() is not None
HAVE_QEMU = shutil.which("qemu-system-arm") is not None


@pytest.mark.skipif(not (HAVE_ARM_GCC and HAVE_MCA), reason="needs arm-none-eabi-gcc + llvm-mca")
def test_census_and_loop_analysis_end_to_end(tmp_path):
    tc = toolchain.discover(ROOT, need=frozenset({"mca"}))
    report = codegen.codegen_report(tc, out_dir=tmp_path)
    # The S1 contract instructions must be present in the production TU.
    assert report.census.counts.get("smlald", 0) >= 1
    assert report.census.counts.get("smulbb", 0) >= 1
    assert report.census.counts.get("qadd16", 0) >= 1
    kernels = {loop.kernel for loop in report.loops}
    assert {"synth_core_m7", "synth_core_pair_m7", "synth_fade_m7"} <= kernels
    assert not report.failed_regions
    for loop in report.loops:
        assert loop.cycles_per_iter > 0
        assert 0 < loop.ipc <= 2.0  # M7 is dual-issue; IPC cannot exceed 2


@pytest.mark.skipif(not (HAVE_ARM_GCC and HAVE_QEMU), reason="needs arm-none-eabi-gcc + qemu")
def test_qemu_counts_end_to_end_reproducible(tmp_path):
    tc = toolchain.discover(ROOT, need=frozenset({"qemu"}))
    report = counts.measure(tc, out_dir=tmp_path, verify_reproducible=True)
    assert report.reproducible
    assert report.rendered_samples == fixture.default_fixture().total_samples
    process = report.range("spectral_arm32_process")
    assert process is not None and process.insns > 0
    # The synth kernel must dominate the dynamic instruction stream.
    total = sum(r.insns for r in report.ranges)
    assert process.insns / total > 0.5
