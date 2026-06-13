"""Harness tests for spectral_tools.performance.embedded (M7 measurement stack).

Unit tests cover the parsing/extraction logic with canned inputs (no
toolchain needed). Integration tests exercise the real rigs end-to-end and
skip cleanly when the cross toolchain / qemu are absent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from spectral_tools.performance.embedded import (  # noqa: E402
    codegen,
    counts,
    fixture,
    mca_validation,
    memory_model,
    toolchain,
    wcet,
)


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


# --- target flags come FROM cmake at runtime (C-truth rule) ------------------

def test_daisy_flags_are_parsed_from_options_cmake():
    """Flags are read from the Daisy CMake defaults at runtime — Python holds
    no copy. Assert the parse works against the real file and yields a
    plausible cross-compile flag set (values themselves belong to CMake)."""
    flags = toolchain.daisy_target_flags(ROOT)
    assert any(f.startswith("-mcpu=") for f in flags)
    assert any(f.startswith("-mfpu=") for f in flags)
    assert any(f.startswith("-mfloat-abi=") for f in flags)
    assert "-mthumb" in flags
    assert any(re.fullmatch(r"-O.+", f) for f in flags)


def test_daisy_flag_parse_fails_loudly_on_contract_move(tmp_path):
    fake_root = tmp_path
    cmake_dir = fake_root / "spectral_engine/cmake"
    cmake_dir.mkdir(parents=True)
    (cmake_dir / "options.cmake").write_text(
        'set(SPECTRAL_DAISY_CPU "-mcpu=cortex-m7" CACHE STRING "x")\n',
        encoding="utf-8",
    )
    with pytest.raises(toolchain.ToolchainError, match="SPECTRAL_DAISY_FPU"):
        toolchain.daisy_target_flags(fake_root)


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


def test_extract_keeps_full_body_with_continue_style_double_backedge():
    # A `continue` produces two back-edges to the same loop-top label. The
    # extractor must keep the span reaching the FARTHEST back-edge, not the
    # truncated early one (else mca models a partial loop body).
    asm = """\
# LLVM-MCA-BEGIN k
.L6:
\tldr r0, [r1], #4
\tcmp r0, #0
\tbne .L6
\tadd r2, r2, r0
\tcmp r1, r3
\tbne .L6
# LLVM-MCA-END
"""
    loops = codegen.extract_loop_bodies(asm)
    assert len(loops) == 1
    # Must include the instructions AFTER the early back-edge, up to the last.
    assert "\tadd r2, r2, r0" in loops[0].instructions
    assert loops[0].instructions[-1] == "\tbne .L6"


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


# --- mca validation set (P3) ------------------------------------------------

def test_validation_cases_cite_real_facts_with_sane_expectations():
    """Every expectation must rest on a recorded, sourced fact — an uncited
    number is exactly the 'heuristic presented as measurement' bug class."""
    assert len(mca_validation.CASES) >= 8
    for case in mca_validation.CASES:
        assert case.body, case.name
        assert case.expected_body_cycles > 0, case.name
        assert case.facts, f"{case.name} cites no facts"
        for fid in case.facts:
            assert fid in mca_validation.FACTS, f"{case.name} cites unknown fact {fid}"
            assert f"[{fid}" in case.derivation or fid in case.derivation, (
                f"{case.name}: derivation does not reference fact {fid}"
            )
    for fid, fact in mca_validation.FACTS.items():
        assert fact["source"], fid
        assert fact["verified"] in {"corroborated", "direct"}, fid


def test_validation_report_math():
    case = mca_validation.CASES[0]
    result = mca_validation.CaseResult(case=case, mca_body_cycles=4.4, mca_loop_cycles=6.4)
    assert result.body_delta_rel == pytest.approx((4.4 - case.expected_body_cycles) / case.expected_body_cycles)
    assert result.mca_loop_overhead == pytest.approx(2.0)
    report = mca_validation.ValidationReport(results=(result,), iterations=100)
    assert report.max_abs_body_delta == pytest.approx(abs(result.body_delta_rel))
    # overhead 2.0 is exactly the hand upper bound -> zero excess divergence
    assert report.mean_backedge_divergence == pytest.approx(0.0)
    doc = report.as_dict()
    assert doc["summary"]["max_abs_body_delta_rel"] == round(report.max_abs_body_delta, 4)
    assert doc["cases"][0]["facts"] == list(case.facts)


# --- memory model (P4) ------------------------------------------------------

def test_memory_constants_parse_from_the_c_ssot():
    """C-truth rule: the model parses daisy_seed_sdram.h + the rig ldscript at
    runtime and fails loudly when the contract moves — no Python copies."""
    c = memory_model.load_constants(ROOT)
    for key in memory_model._REQUIRED_DEFINES:
        assert key in c.defines, key
    # The chosen-performance timing set: TRP/TRCD are the datasheet-derived
    # values, NOT libDaisy's debug-era 16/10.
    assert c.defines["SPECTRAL_DAISY_SDRAM_TRP"] < memory_model.STOCK_LIBDAISY_TICKS["TRP"]
    assert c.defines["SPECTRAL_DAISY_SDRAM_TRCD"] < memory_model.STOCK_LIBDAISY_TICKS["TRCD"]
    # Marker address comes from the ldscript MARKER region.
    assert c.marker_addr == 0x60F00000
    # Derived per-line costs: row miss > row hit > pure bandwidth; the chosen
    # set must beat the stock set on the row-miss path (the point of choosing).
    hit, miss = c.line_fill_cycles(True), c.line_fill_cycles(False)
    assert 0 < hit < miss
    assert hit > c.linefill_beats * c.sdclk_per_cpu_cycle
    assert miss < c.line_fill_cycles_stock(False)
    # The model assumption stays labeled.
    assert "ASSUMPTION" in memory_model.BRIDGE_PROVENANCE


def test_memory_constants_fail_loudly_on_contract_move(tmp_path):
    fake_root = tmp_path
    hdr = fake_root / "api/daisy_seed"
    hdr.mkdir(parents=True)
    (hdr / "daisy_seed_sdram.h").write_text(
        "#define SPECTRAL_DAISY_CPU_HZ 400000000u\n", encoding="utf-8")
    with pytest.raises(memory_model.MemoryModelError, match="SPECTRAL_DAISY_SDCLK_HZ"):
        memory_model.load_constants(fake_root)


def test_dcache_sim_compulsory_capacity_and_write_allocate():
    c = memory_model.load_constants(ROOT)
    sim = memory_model.DCacheSim(c)
    line = c.line_bytes
    cache_bytes = c.defines["SPECTRAL_DAISY_DCACHE_BYTES"]
    base = 0x60000000

    # Sequential cold reads: every line a compulsory miss, re-read: all hits.
    n_resident = (cache_bytes // line) // 2   # half the cache, conflict-free
    for i in range(n_resident):
        sim.access(memory_model._Access(False, base + i * line, 4))
    assert sim.read_misses == n_resident and sim.read_hits == 0
    for i in range(n_resident):
        sim.access(memory_model._Access(False, base + i * line, 4))
    assert sim.read_hits == n_resident

    # Streaming twice through 4x the cache: capacity misses on the second pass.
    sim2 = memory_model.DCacheSim(c)
    n_stream = (cache_bytes // line) * 4
    for _ in range(2):
        for i in range(n_stream):
            sim2.access(memory_model._Access(False, base + i * line, 4))
    assert sim2.read_misses == 2 * n_stream

    # Write-allocate: a partial-line store miss fills; a full-line aligned
    # store stream switches to no-fill (dynamic read allocate class).
    sim3 = memory_model.DCacheSim(c)
    sim3.access(memory_model._Access(True, base, 4))
    assert sim3.write_miss_fills == 1
    sim4 = memory_model.DCacheSim(c)
    for i in range(4):
        sim4.access(memory_model._Access(True, base + i * line, line))
    assert sim4.write_miss_no_fill == 4 and sim4.write_miss_fills == 0


def test_row_tracker_sequential_cadence():
    c = memory_model.load_constants(ROOT)
    rows = memory_model.RowTracker(c)
    row_bytes = c.defines["SPECTRAL_DAISY_SDRAM_ROW_BYTES"]
    misses = sum(0 if rows.access(addr) else 1
                 for addr in range(0x60000000, 0x60000000 + 8 * row_bytes, 64))
    # Sequential stream: exactly one row-open per row_bytes span.
    assert misses == 8


def test_analyze_trace_splits_epochs_and_bounds(tmp_path):
    c = memory_model.load_constants(ROOT)
    line = c.line_bytes
    trace = tmp_path / "trace.txt"
    rows = [
        f"S {0x60000000:x} 4",            # sdram store miss -> fill
        f"L {0x20000000:x} 4",            # dtcm: counted, never priced
        f"S {c.marker_addr:x} 4",         # epoch boundary
        f"L {0x60000000 + line:x} 4",     # sdram read miss -> fill
        f"L {0x60000000 + line:x} 4",     # hit
        f"L {0x08000000:x} 4",            # flash-class
        f"S {c.marker_addr:x} 4",
    ]
    trace.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rep = memory_model.analyze_trace(trace, c)
    assert len(rep.blocks) == 2
    b0, b1 = rep.blocks
    assert b0.sdram_accesses == 1 and b0.dtcm_accesses == 1
    assert b1.sdram_accesses == 2 and b1.flash_accesses == 1
    total_fills = rep.total.fills_row_hit + rep.total.fills_row_miss
    assert total_fills == 2
    for b in rep.blocks:
        assert b.stall_bandwidth <= b.stall_serial
    doc = rep.as_dict()
    assert doc["n_blocks"] == 2
    assert "ASSUMPTION" in doc["constants"]["bridge_cpu_cycles"]["provenance"]
    # The report must show the chosen-vs-stock comparison so the win is visible.
    derived = doc["derived_per_line_cycles"]
    assert derived["linefill_row_miss"] < derived["stock_libdaisy_comparison"]["linefill_row_miss"]


# --- WCET synthesis (P5) ----------------------------------------------------

def _synthetic_wcet(active=64, scan=1024, block=256) -> wcet.WcetReport:
    c = memory_model.load_constants(ROOT)
    inputs = wcet.WcetInputs(
        worst_cyc_per_voice_sample=24.0,
        worst_kernel="synth_fade_m7/.L624",
        sustain_insns_per_voice_sample=15.8,
        measured_insns_per_block=58_800.0,
        measured_voice_samples_per_block=2304.0,
        fixture_peak_active=9,
        code_lines=310,
        data_lines=4500,
    )
    return wcet.WcetReport(inputs=inputs, constants=c,
                           active=active, scan_segments=scan, block=block)


def test_wcet_terms_are_bounds_with_provenance():
    rep = _synthetic_wcet()
    # T1 is exact arithmetic over the validated per-voice-sample rate.
    assert rep.t1_synth_cycles == 64 * 256 * 24.0
    # Residual subtracts at the cheapest loop rate and scales by active ratio:
    # both choices over-estimate (documented), so the term must be positive.
    assert rep.residual_insns > 0
    assert rep.t2_residual_cycles > rep.residual_insns  # CPI bound >= 1
    assert rep.t3_cold_memory_cycles > 0
    doc = rep.as_dict()
    assert doc["wcet_block_cycles"] == round(rep.wcet_cycles, 0)
    for term in doc["terms"].values():
        assert "provenance" in term and term["cycles"] >= 0
    assert "BOUND" in doc["provenance"]
    assert doc["constants_source"] == memory_model.SDRAM_HEADER_RELPATH


def test_wcet_monotonic_in_parameters():
    base = _synthetic_wcet(active=64, scan=1024)
    more_active = _synthetic_wcet(active=128, scan=1024)
    more_scan = _synthetic_wcet(active=64, scan=4096)
    assert more_active.wcet_cycles > base.wcet_cycles
    assert more_scan.wcet_cycles > base.wcet_cycles


def test_wcet_unroll_map_names_real_kernels():
    expected = {"synth_core_m7", "synth_core_pair_m7", "synth_fade_m7"}
    assert {k.split("/")[0] for k in wcet.SAMPLES_PER_ITER} == expected
    # Bound constants must be documented as bounds where they are not measured.
    import inspect

    src = inspect.getsource(wcet)
    for name in ("RESIDUAL_CPI_BOUND", "SCAN_INSNS_PER_CHECK_BOUND"):
        assert f"[bound:" in src and name in src


# --- counts table parsing ---------------------------------------------------

CANNED_COUNTS = """\
# [measured: qemu-tcg dynamic counts] — not cycles
range                                 insns        loads       stores     load_bytes    store_bytes     lines32B
<other>                              498145        74155        24258         460452          94394         1201
spectral_arm32_process              3763544      1097748       353301        4566738        1652298         4587
# data bytes by region: code=508272 ssram23=6324410 bulk60=0 other=8
# unique 32B lines by region: code=310 ssram23=4496 bulk60=0 other=1
"""


def test_parse_counts_table():
    ranges, regions, region_lines = counts._parse_counts(CANNED_COUNTS)
    names = {r.name for r in ranges}
    assert names == {"<other>", "spectral_arm32_process"}
    process = next(r for r in ranges if r.name == "spectral_arm32_process")
    assert process.insns == 3763544
    assert process.load_bytes == 4566738
    assert process.lines32 == 4587
    assert regions == {"code": 508272, "ssram23": 6324410, "bulk60": 0, "other": 8}
    assert region_lines == {"code": 310, "ssram23": 4496, "bulk60": 0, "other": 1}


def test_parse_counts_rejects_garbage():
    with pytest.raises(counts.CountsError, match="no count rows"):
        counts._parse_counts("nothing useful here\n")


# --- measurement matrix ------------------------------------------------------

def test_matrix_targets_and_probes():
    from spectral_tools.performance import matrix

    names = {profile.name for profile in matrix.TARGETS}
    assert {"desktop", "simulate", "m7", "daisy"} <= names
    with pytest.raises(KeyError, match="unknown measurement target"):
        matrix.target("nonsense")

    probed = matrix.probe_matrix()
    for target_name, entry in probed.items():
        assert entry["instruments"], f"{target_name} has no instruments"
        for row in entry["instruments"]:
            assert row["availability"] in {"available", "unavailable", "planned"}
            assert row["provenance"] in {"measured", "modeled"}
            assert row["kind"] in {"timing", "counters", "memory", "counts", "model"}


def test_matrix_m7_instruments_probe_real_toolchain():
    """On this machine the m7 instruments should probe available (toolchain
    + qemu + mca installed); elsewhere they must degrade to unavailable with
    an actionable hint, never raise."""
    from spectral_tools.performance import matrix

    entry = matrix.probe_matrix()["m7"]
    by_name = {row["instrument"]: row for row in entry["instruments"]}
    assert by_name["qemu-counts"]["availability"] in {"available", "unavailable"}
    assert by_name["mca-census"]["availability"] in {"available", "unavailable"}
    for row in entry["instruments"]:
        if row["availability"] == "unavailable":
            assert row["detail"], "unavailable instrument must say how to get it"


# --- integration (skipped when toolchain absent) -----------------------------

def _have_toolchain(*need: str) -> bool:
    try:
        toolchain.discover(ROOT, need=frozenset(need))
    except toolchain.ToolchainError:
        return False
    return True


HAVE_ARM_GCC = _have_toolchain()        # requires a newlib-capable toolchain
HAVE_MCA = _have_toolchain("mca")
HAVE_QEMU = _have_toolchain("qemu")


@pytest.mark.skipif(not HAVE_MCA, reason="needs newlib arm-none-eabi-gcc + llvm-mca (m7-bootstrap)")
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


@pytest.mark.skipif(not HAVE_MCA, reason="needs newlib arm-none-eabi-gcc + llvm-mca (m7-bootstrap)")
def test_mca_validation_against_hand_derived_timings(tmp_path):
    """The P3 contract: CortexM7Model body throughput must stay within 10% of
    the hand-derived TRM/community expectations on every microbench, and the
    back-edge divergence must not exceed the hand range by more than 1.5
    cycles. At landing (llvm 21.1.8) the body deltas were <= 1% and the
    divergence 0.0; a failure here means the shipped mca model drifted and the
    kernel cycles/iter error bound needs re-derivation."""
    tc = toolchain.discover(ROOT, need=frozenset({"mca"}))
    report = mca_validation.validate(tc, out_dir=tmp_path)
    assert len(report.results) == len(mca_validation.CASES)
    for result in report.results:
        assert abs(result.body_delta_rel) <= 0.10, (
            f"{result.case.name}: mca {result.mca_body_cycles} vs hand "
            f"{result.case.expected_body_cycles}"
        )
    assert report.mean_backedge_divergence <= 1.5


@pytest.mark.skipif(not HAVE_QEMU, reason="needs newlib arm-none-eabi-gcc + qemu")
def test_fft_probe_measures_cmsis_rfft(tmp_path):
    """F-stream step 1: the vendored CMSIS-DSP inverse RFFT-512 measures at a
    stable instruction count on the rig. Band: 30K-60K insns (measured 43,910
    at v1.16.2 / xPack 15.2.1) — a move outside it means the vendored DSP lib
    or the toolchain changed the F2 crossover input and the pricing in
    M7_PERF_MODEL_PLAN must be re-derived."""
    from spectral_tools.performance.embedded import fft_probe

    tc = toolchain.discover(ROOT, need=frozenset({"qemu"}))
    r = fft_probe.probe(tc, out_dir=tmp_path)
    assert 30_000 < r.insns_per_fft < 60_000
    assert "arm_rfft_q31" in r.compute_symbols
    assert "arm_radix4_butterfly_inverse_q31" in r.compute_symbols
    doc = r.as_dict()
    assert doc["cycles_per_fft_band"][0] < doc["cycles_per_fft_band"][1]
    assert "measured: qemu-tcg" in doc["provenance"]


@pytest.mark.skipif(not (HAVE_MCA and HAVE_QEMU),
                    reason="needs newlib arm-none-eabi-gcc + llvm-mca + qemu")
def test_wcet_end_to_end_composes_live_stack(tmp_path):
    """P5 contract: the WCET bound derives from LIVE stack outputs (no cached
    numbers), every kernel passes the unroll-drift guard, and the default
    64-active/1024-scan scenario stays a meaningful fraction of the block
    budget (a bound near 0% or far past 1000% means a term broke)."""
    tc = toolchain.discover(ROOT, need=frozenset({"mca", "qemu"}))
    rep = wcet.wcet(tc, out_dir=tmp_path)
    doc = rep.as_dict()
    assert rep.inputs.worst_cyc_per_voice_sample > 0
    assert 0.05 < doc["budget_fraction"] < 10.0
    # The bound must exceed the pure synthesis term (T2/T3 are real charges).
    assert rep.wcet_cycles > rep.t1_synth_cycles


@pytest.mark.skipif(not HAVE_QEMU, reason="needs newlib arm-none-eabi-gcc + qemu (m7-bootstrap)")
def test_memory_model_end_to_end_over_real_trace(tmp_path):
    """P4 contract on the default fixture: the trace splits into init/load +
    one epoch per block (+ epilogue); the tiny 9-segment store becomes
    D-cache resident during load, so EVERY steady-state block models zero
    SDRAM stalls — the fills all sit in epoch 0. A steady-state epoch with
    nonzero stalls means the placement mirror or the cache model regressed."""
    tc = toolchain.discover(ROOT, need=frozenset({"qemu"}))
    trace_path = tmp_path / "trace.txt"
    counts.measure(tc, out_dir=tmp_path, verify_reproducible=False,
                   trace_path=trace_path)
    rep = memory_model.analyze_trace(trace_path)

    spec = fixture.default_fixture()
    n_render_blocks = spec.total_samples // spec.block_samples
    assert len(rep.blocks) == n_render_blocks + 2  # init/load + blocks + epilogue

    fills = rep.total.fills_row_hit + rep.total.fills_row_miss
    assert fills > 0, "the bulk-placed segment store must cause linefills"
    b0 = rep.blocks[0]
    assert b0.fills_row_hit + b0.fills_row_miss == fills, "all fills belong to init/load"
    for b in rep.blocks[1:]:
        assert b.stall_serial == 0.0
    # The hot path is DTCM-class by placement: it must dominate the traffic.
    assert rep.total.dtcm_accesses > 100 * rep.total.sdram_accesses


@pytest.mark.skipif(not HAVE_QEMU, reason="needs newlib arm-none-eabi-gcc + qemu (m7-bootstrap)")
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

@pytest.mark.skipif(not HAVE_ARM_GCC, reason="needs arm-none-eabi-gcc (m7-bootstrap)")
def test_dormant_dma_branch_still_compiles(tmp_path):
    """No build target sets SPECTRAL_HAS_DMA (the BSP hooks need a board), so
    nothing else ever compiles that branch of the arm32 kernel. Dormant code
    that cannot rot silently: cross-compile the TU with the DMA configuration
    forced on (syntax/type check only — the BSP externs resolve at link)."""
    import subprocess

    tc = toolchain.discover(ROOT)
    tu = ROOT / "spectral_engine/arch/arm/spectral_synth_arm32.c"
    arm_dir = ROOT / "spectral_engine/arch/arm"
    result = subprocess.run(
        [tc.arm_gcc, *tc.cflags(extra_includes=(arm_dir,)),
         "-DSPECTRAL_HAS_DMA=1", "-DSPECTRAL_ARM32_DMA_BUFFER_DTCM=1",
         "-fsyntax-only", str(tu)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"dormant DMA branch no longer compiles:\n{result.stderr[-3000:]}")
