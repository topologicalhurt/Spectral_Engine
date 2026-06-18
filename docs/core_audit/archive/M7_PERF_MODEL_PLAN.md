# M7 Performance Model — Campaign 3 / S1 measurement stack

> **ARCHIVED 2026-06-18.** P0–P6 complete; the no-hardware Cortex-M7 perf-model stack ships and the perf-gate is green. The live contract is the code/baseline it produced (census/qemu/cycles/memory/wcet); AI.md cites this doc for provenance.

Goal (REVIEWER_HANDOFF §S1): the most accurate Cortex-M7 performance model obtainable
without hardware — instruction cycles, cache/prefetch behavior, bytes moved — with
**measured** quantities strictly separated from **modeled** ones. The existing
`runtime/spectral_perf_model.*` profile constants are uncalibrated heuristics; they are
replaced by (or re-derived from) this stack, or the sim/perf-model pair is deprecated (P5).

## Decision record — approaches evaluated (P0, closed)

| Candidate | Verdict | Why |
|---|---|---|
| gem5 | rejected | no maintained Cortex-M profile; M7 in-order dual-issue + TCM/cache + H7 interconnect would be a from-scratch model with nothing to calibrate against |
| OVPsim / Imperas / Arm Cycle Models | rejected | commercial licensing; not obtainable here |
| Renode | rejected | functional emulation; timing is virtual-time pacing, not a cycle model |
| QEMU (mps2-an500, Cortex-M7) + TCG plugin | **adopted (counts only)** | exact dynamic instruction streams and load/store traces of the real ELF; TCG has *no* timing fidelity by design — it is the measured-counts layer, never a cycle source |
| llvm-mca, `-mcpu=cortex-m7` | **adopted (pipeline layer)** | Arm-contributed `CortexM7Model` (LLVM D91355; drove llvm-mca in-order support). Dual-dispatch in-order pipeline simulation per loop body. Blind spots: perfect memory, no branch effects — bounded by Layer 3 and the validation protocol |
| Calibrated analytical memory model | **adopted (memory layer)** | DTCM/AXI/SDRAM latencies + 32B-line cache model, calibrated against ST AN4891 (measured H7 memory-path numbers) and the Daisy H750 FMC/SDRAM timings; consumes Layer-1 measured address traces, not assumptions |

## The layered stack

- **Layer 0 — static, measured:** real-TU codegen census. `arm-none-eabi-gcc -mcpu=cortex-m7
  -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -O3 -ffreestanding` over the actual backend TUs
  (freestanding libc stub headers; brew toolchain has no newlib). Outputs: per-kernel
  disassembly, instruction census, proof the intended ops (SMLALD, SMULBB, QADD16, SSAT)
  are emitted. Rig proven; first run found and fixed a real on-target build break
  (`__smulbb` is not an ACLE intrinsic — the `__ARM_FEATURE_DSP` branch never compiled
  under GNU ARM GCC).
- **Layer 1 — dynamic, measured:** QEMU 11 `mps2-an500` + a purpose-built TCG plugin
  (`qemu-plugin.h` ships with the brew build). A freestanding runner (vector table, minimal
  crt0, mps2-an500 ldscript, semihosting exit via `bkpt 0xab`; local memcpy/memset) executes
  the real `spectral_arm32_init/load/process` on deterministic fixtures. Outputs: exact
  retired-instruction counts attributed per function PC-range, exact load/store counts,
  bytes moved, address streams → cache-line working sets. Counts are bit-deterministic
  run-to-run or the rig is broken. Fallback if darwin plugin loading fails: `-d exec`
  trace + offline counter (slower, same numbers).
- **Layer 2 — modeled (validated):** llvm-mca `CortexM7Model` steady-state cycles/iteration
  for each hot loop body extracted from Layer 0. Validation protocol: a microbench set of
  hand-analyzable loops (pure ALU dual-issue, MAC chains, load-use stalls) checked against
  the M7 TRM issue rules and community-measured M7 timings; per-loop delta reported. mca
  numbers are always labeled `[modeled: mca/CortexM7Model]`.
- **Layer 3 — modeled (calibrated):** memory-stall model. Inputs: Layer-1 address streams +
  placement (DTCM zero-wait dual-port / AXI-SRAM via 64-bit AXIM / SDRAM via FMC), 32B
  lines, 4-way D-cache, write-buffer behavior. Latency table calibrated against AN4891
  measured tables + Daisy H750 SDRAM timing config; each constant carries its provenance.
- **Synthesis:** cycles/block = Σ(L2 steady-state × L1 measured iterations) + L3 stalls over
  L1 measured traffic, with an explicit error bound. WCET reported separately (max active
  voices, cold cache, SDRAM worst path). Every reported number is tagged `[measured]` or
  `[modeled:<source>]`. A heuristic presented as a measurement is a bug.

## QEMU fidelity contract (what the counts oracle is — and is not)

QEMU 11 `mps2-an500`/cortex-m7 is an **ISA-level oracle for architecturally-executed
instruction and memory-access counts and IEEE-754 FP bit-exactness. It models zero
microarchitecture** (no caches, no cycles, no DWT). Verified, not assumed:

- The CPU model implements ARMv7E-M + DSP + FPv5-D16 double; MVFR0/1/2 ID values are
  bit-identical to the Cortex-M7 TRM (DDI 0489F) reset values `[sourced: qemu
  target/arm/tcg/cpu-v7m.c vs TRM Table 8-2]`.
- Plugin exec callbacks fire exactly once per **architecturally executed** instruction,
  including predicated-false IT-block members (measured: two byte-identical IT loops,
  always-true vs always-false predicate, both count 60007); mem callbacks fire only for
  **architecturally performed** accesses (predicated-false LDR produced zero) `[measured]`.
  This matches the ARM definition — hardware also "executes" cond-fail instructions.
- FP: FPSCR resets to 0 (FZ=0, DN=0, RN) exactly as hardware FPDSCR; softfloat honors
  FZ/DN/rounding per spec (measured on subnormal cases); IEEE + identical FPSCR ⇒
  identical bits `[measured + inferred]`.
- Caches absent by design: SCB cache ops are NOPs, CLIDR/CCSIDR read 0 — counts contain
  zero cache effects, which is why cycles/caches are separate layers (P3/P4).

The contract holds under four conditions (all enforced or documented in the rig):
1. **FPSCR is pinned** — the runner writes FPSCR=0 at reset, matching the Daisy default
   (`SPECTRAL_DAISY_SAFE_MATH=ON` ⇒ no `-ffast-math` ⇒ no crtfastmath FZ=1). If safe-math
   is ever disabled on hardware, double-precision parity claims must restate FPSCR.
2. Kernels stay RAM-only, fault-free, interrupt-free (M-profile exception stacking is
   plugin-invisible; MMIO-touching code risks exec-cb double-fire on TB restarts).
3. Semihosting, cache-init (CCSIDR-driven loops degenerate on QEMU), and DWT/CYCCNT reads
   stay outside counted ranges (DWT is RAZ/WI on QEMU and would change control flow).
4. Counts are never read as cycles or cache behavior.

Known platform deltas, all counts-irrelevant under the conditions above: mps2-an500
memory map (placement is ldscript-controlled), no FMC/SDRAM controller, bitband present
on QEMU but absent on real M7 (QEMU strictly more permissive), MPU 8 regions vs 16.

## Hot kernels under the model

`synth_core_m7` (4×-unrolled coupled-step + MAC), `synth_core_pair_m7` (SMLALD dual-voice),
`synth_fade_m7` (fade ramp), `spectral_coupled_step` (4×SMULL/SMLAL class per sample, inlined
into all three), plus the block-level scan/activation/prune in `spectral_arm32_process` and
the segment-load/DMA path.

## Phases

- **P0 — survey + decision.** DONE (this doc is the record).
- **P1 — codegen rig committed.** Compiles the real TUs, extracts the hot-loop bodies,
  emits the instruction census (now `performance/embedded/codegen.py`; the original
  shell rig is superseded). Done-when: one command regenerates the census for the four
  hot kernels.
- **P2 — QEMU counts rig.** Freestanding runner + TCG plugin + driver script; deterministic
  fixtures shared with `arm_core_test` so correctness and counts come from the same inputs.
  Done-when: real `spectral_arm32_process` runs under qemu-system-arm and per-kernel
  instruction/memory counts are reproducible exactly.
- **P3 — pipeline layer.** Automated mca over extracted loop bodies + the validation
  microbench set with TRM cross-check. Done-when: cycles/iteration per kernel with stated
  validation delta. **CLOSED (pass 250)** — see Status.
- **P4 — memory layer.** Line-footprint analyzer over L1 traces; AN4891-calibrated latency
  table; SDRAM segment-stream/DMA model. Done-when: per-block stall estimate with per-constant
  calibration provenance. **CLOSED (pass 251)** — see Status.
- **P5 — integration.** Re-derive or retire `spectral_perf_model.c` profiles from the stack
  (decision point: the sim earns its keep here or is deprecated); WCET report; feeds S4
  benchmark redesign. **CLOSED (pass 253)** — see Status.
- **P6 — application.** Re-evaluate S1 optimization candidates against the model: loop-nest /
  data-layout inversion, SMLALD coverage beyond full-sustain pairs, ITCM code placement,
  and the F2 oscillator-vs-IFFT embedded crossover. Maintainer emphasis (2026-06-12):
  the bar is *meaningful saturation* of the execution units the silicon has — FPU, both
  ALUs, the MAC pipe, the dual-issue slots, the cache/TCM hierarchy — treated as an
  embedded platform first, not a ported desktop loop. **CMSIS / CMSIS-DSP and the
  embedded SIMD treatment are explicitly in scope here** (the codebase has not
  thought them through together yet; SIMDe is host-only by design, the embedded
  dispatch has no CMSIS-DSP wiring — see OSCILLATOR_BACKEND_CONTRACT_PLAN's open
  CMSIS item). End state: the architecture/algorithm choice is settled (mirroring
  the desktop path's best design, ported to ARM), so that compiler-directive
  tinkering and hand-written assembly become a *meaningful* final consideration
  rather than premature effort — P3's validated per-loop model + P4's memory layer
  are what make that judgment defensible without hardware.

## Status

- P0 closed. P1 closed (census: `smlald`×1, `smulbb`×2, `qadd16`×39, `ssat`×2,
  `smull`×105, `smlal`×37 in the M7 TU).
- Defect found by the rig, fixed: `spectral_smulbb` DSP branch called nonexistent ACLE
  intrinsic `__smulbb`; replaced with widened multiply, SMULBB emission codegen-verified,
  ctest green.
- First Layer-2 numbers `[modeled: llvm-mca/CortexM7Model, perfect memory]`: main
  sustain loop 340 cyc / 16 samples ≈ 21.3 cyc/sample (GCC re-unrolled the 4× body to
  16 samples/iter); SMLALD pair loop 40 cyc / 2 voice-samples = 20.0; scalar tail 18.
  Pairing barely wins under perfect memory — its true win is halved accumulator
  read-modify-write traffic, which only the P4 memory layer prices. Do not draw
  pairing-coverage conclusions (P6) before P4.
- P2 closed: the real kernel runs on qemu-system-arm mps2-an500 with the
  `spectral_counts` TCG plugin; counts verified bit-identical across runs (the
  duplicate-run check is on by default). First Layer-1 numbers `[measured: qemu-tcg]`
  for the 9-voice/16384-sample fixture: `spectral_arm32_process` = 3,763,544 insns,
  4,566,738 B loaded / 1,652,298 B stored ⇒ ~51.0 insns and ~62 ld + ~22 st bytes per
  voice-sample. The q63 accumulator read-modify-write dominates traffic — the measured
  input the A2 loop-nest/data-layout question was missing. Known rig divergences from
  the Daisy target (deliberate, attributed separately): byte-loop memcpy/memset,
  runner-local sinf via the kernel's f64 init sine.
- **Harness home (consolidation pass, 2026-06-10):** the whole stack lives in
  `tools/spectral_tools/performance/embedded/` (toolchain/fixture/codegen/counts +
  `native/` C sources); the original shell scripts are deleted. One CLI:
  `python -m spectral_tools.testing.benchmark_workflow m7-census | m7-counts |
  measure --target m7` (CMake: `m7_census` / `m7_counts`, same family as `bench`).
  The workload fixture is a Python SSOT that generates the C header the runner
  compiles; reports carry the fixture digest and a measured/modeled provenance tag
  on every number. Target flags are parsed from options.cmake at runtime (no copy in
  Python). Harness behavior is itself under test: `pytest tests/tools`. Migration
  verified: kernel-range counts and mca per-loop cycles bit/numerically identical
  to the pre-migration rig.
- **P3 closed (pass 250, 2026-06-12).** The Layer-2 validation protocol ran: a
  9-case microbench set of hand-analyzable loops (`performance/embedded/
  mca_validation.py`, CLI `m7-mca-validate`, pytest-covered) whose expected
  cycles/iteration are derived ONLY from sourced facts — the Cortex-M7 TRM
  DDI 0489F structure statements (2-skeptic-verified + the full TRM now local)
  and the two community DWT measurement sets (quinapalus, STM32H730@480MHz;
  jnk0le, STM32H743 — both re-verified first-party at source during pass 250;
  ARM publishes no per-instruction M7 cycle table, which is itself sourced).
  Covered shapes: ALU dual-issue + dependent-chain serialization, banked
  load/load pairing, load-use forwarding, single-MAC-pipe throughput,
  SMLAL/SMLALD accumulator chaining (the `spectral_coupled_step` /
  `synth_core_pair_m7` shapes), single-store-channel, load+store pairing.
  **Result: CortexM7Model body-throughput delta ≤1% on 9/9 cases** (llvm-mca
  21.1.8). One known systematic: the model has no branch folding, charging
  ~1–2 cyc/iter for a predicted-taken back-edge that real M7 folds to ~0–1
  [jnk0le] — an *upper* bias on loop numbers, bounded per kernel below.
  Validation deltas are model-vs-derivation distances, not hardware truth
  (no board; fidelity contract unchanged).
- **Per-kernel cycles/iteration with stated validation delta (P3 done-when),**
  current census (xPack 15.2.1, llvm-mca 21.1.8) `[modeled: mca/CortexM7Model,
  perfect memory; body-throughput validated ≤1%; back-edge bias ≤+2 cyc/iter]`:
  | kernel loop | cyc/iter | per voice-sample | back-edge bias bound |
  |---|---|---|---|
  | `synth_core_m7` sustain (16-sample) | 340 | 21.3 | ≤0.6% |
  | `synth_core_pair_m7` SMLALD pair | 40 | 20.0 | ≤5% |
  | `synth_core_m7` scalar tail | 18 | 18 | ≤11% |
  | `synth_fade_m7` fade ramp | 22 | 22 | ≤9% |
  Memory effects are NOT in these numbers — that is P4's layer; do not compare
  them against wall-clock or QEMU counts without it.
- **P4 closed (pass 251, 2026-06-12).** Layer 3 lives in
  `performance/embedded/memory_model.py` (CLI `m7-stalls`; matrix instruments
  `mca-validation`, `memory-stalls`): the QEMU plugin's program-order address
  trace replays through the Daisy placement mirror (rig change: `segbuf` now in
  `.bulk_bss` @0x60000000; per-block `.bulk_marker` stores split the trace into
  init/load + per-block epochs) → a 16 KB 4-way 32B-line WBWA D-cache model
  (incl. a dynamic-read-allocate approximation for full-line write streams) +
  a per-bank open-row tracker → stall **bounds** per epoch. Bounds, not points:
  the M7 is non-blocking (2 data linefill buffers + merging store buffer [TRM]),
  so N misses cost between the bandwidth bound and the serial-latency bound.
  **Every constant carries provenance**: libDaisy-as-shipped FMC config
  (CAS=3, 32-bit bus, SDCLK=HCLK/2) + clock tree (CPU 400 / AXI 200 / SDCLK
  100 MHz → 4 CPU cyc per SDCLK), TRM structure facts, AN4891 Rev 1 measured
  anchors (DTCM zero-latency p17; D-cache 16 KB p5; flash+I-cache ≈ ITCM ×1.01
  Table 7; cached-SDRAM FFT data ×1.19 Table 6 as the order-of-magnitude
  cross-check). One labeled ASSUMPTION: the AXIM→FMC bridge overhead
  (12 CPU cyc) — uncalibrated without a board. Derived per-line costs:
  row-hit linefill **56 cyc**, row-miss **160 cyc**, writeback 44 cyc.
- **Finding (upstreamable): libDaisy SDRAM timing is ~4× conservative.**
  `sdram.cpp` ships `RPDelay=16`, `RCDDelay=10` (comments admit "started at 2")
  vs the AS4C16M32MSA-6 datasheet minima (~3+3 SDCLK at 100 MHz). As shipped, a
  row-miss linefill costs 160 CPU cyc; at datasheet timings ~70. Also: the
  shipped `SelfRefreshTime` (tRAS) of 4 ticks = 40 ns is **below** the 48 ns
  minimum libDaisy's own comment quotes — out of spec, not just slow.
- **Pass 252 — constants centralized; best-performance timing CHOSEN
  (maintainer directive).** `api/daisy_seed/daisy_seed_sdram.h` is now the ONE
  SSOT for the Daisy memory-system numbers (clock tree, FMC timing, SDRAM and
  cache geometry; per-value provenance in-header). The Python model parses it
  at runtime (C-truth, fail-loud), the counts plugin takes `lineshift=` from
  it, the trace marker address is parsed from the rig ldscript. The CHOSEN
  FMC set is the datasheet-derived minima (TRP/TRCD 2/2, TWR 2, TRAS 5,
  TRC 7; CAS kept at 3 — a chip mode-register reload is not worth an
  untestable 4-cycle gain) and `daisy_spectral_init()` programs it into
  FMC_SDTR1 (`SPECTRAL_SDRAM_STOCK_TIMINGS` opts out; **needs on-target
  verification when a board exists**). Model under the chosen set: row-miss
  linefill **72 cyc** (was 160 stock); the report keeps the stock comparison
  row so the win stays visible.
- **Fixture result + the scaling input P6/F needs.** For the 9-voice fixture
  the whole segment store is 5 cache lines: all linefills land in the init/load
  epoch, **steady-state per-block SDRAM stalls = 0**, and the dominant 1.5 M
  DTCM-class accesses are zero-wait by placement — i.e. the mca (L2) numbers
  ARE the block cost for DTCM-resident workloads, as designed. The model's
  marginal prices for non-resident (streaming) segment stores, at 14–16 B per
  `SpectralSegmentQ15` ≈ 2 segments/line: **per-block scan ≈ 14 N (bandwidth
  bound) to 25 N (serial bound) cycles for N streamed segments** (row misses
  amortize at 1 per 2 KB). At N=1000 that is ~14–25 K cyc/block on top of
  ~49 K cyc compute (9 active voices × 256 samples × 21.3) — the magnitude that
  prices the A2 loop-nest question, SDRAM-side DMA double-buffering, and the F2
  osc-vs-IFFT crossover. Boundaries stated honestly: the DMA segment-stream
  variant is priced by the same bandwidth constants (no separate code path
  pretends otherwise); flash-class data reads are counted but not priced
  (cached + AN4891 ×1.01 anchor); I-side fetch is outside the data trace.
- **P5 closed (pass 253, 2026-06-12).** Decision: **RETIRED, not re-derived.**
  `runtime/spectral_perf_model.c/h` deleted on evidence — every constant was
  uncalibrated (confidence/pessimism numerology) and the op-cost taxonomy
  priced a kernel shape that no longer exists (per-sample LUT gather, replaced
  by the coupled-form oscillator in pass 236; 4×-unroll loop term vs the real
  16×). Re-deriving constants could not fix a wrong structure. The sim KEEPS
  its real value: it runs the REAL `spectral_arm32_process` and reports only
  MEASURED workload counters (`voice_samples` — renamed from the stale
  lut/mac/phase trio — scan checks, activations, `accum_rw_words`, on-device
  DWT peak-block) plus budget arithmetic; the host never fabricates cycles
  and the report points at this stack for projections.
  **WCET (the P5 done-when):** `performance/embedded/wcet.py` (CLI `m7-wcet`)
  composes the live layers into a parametric per-block upper BOUND:
  T1 synthesis = active × block × worst validated cyc/voice-sample (+ the P3
  back-edge bias worst case); T2 non-loop residual = measured fixture insn
  profile scaled conservatively × CPI bound [bound: ≤2, justified] + per-scan
  worst-case mispredicts [measured-community: 8 cyc]; T3 cold memory =
  working-set + scan lines × the P4 row-miss serial fill (SDRAM price
  upper-bounds flash). An unroll-drift guard fails loudly if codegen moves.
  Reference scenario: **64 active voices + 1024-segment scan + 256-sample
  block ≈ 1.02 M cycles = 48% of the 48 kHz block budget @400 MHz** —
  64 worst-case voices are schedulable with margin. WCET excludes init/load
  and assumes the Daisy DTCM placement contract.
- **P6 decided (pass 254, 2026-06-12) — the model applied to the open
  optimization candidates.** Three decisions, each on the stack's numbers:
  - **D1 loop-nest / data-layout inversion (A2): DECLINED on data.** The
    motivating win was halved q63-accumulator memory traffic
    (`accum_rw_words`, measured). The accumulator is DTCM-resident per the
    placement contract and DTCM is zero-wait dual-ported [TRM / AN4891 p17]
    — the traffic difference prices to **zero stall cycles** (P4). The
    accumulator ld/st issue slots are already inside the validated
    21.3 cyc/voice-sample. Inverting the nest would *add* per-sample reload
    traffic of per-voice oscillator state. No win exists at any voice count
    while the accumulator stays DTCM-resident.
  - **D2 SMLALD pairing extension beyond full-sustain: DECLINED.** Validated
    rates: pair loop 20.0 vs sustain 21.3 cyc/voice-sample = **6.1% on the
    paired fraction only**; the traffic half of the win prices to zero (D1).
    Extending pairing into fade-phase/partial-overlap voices buys ≤6% of a
    fraction of synthesis time for real bookkeeping complexity at activation.
    The existing full-sustain pairing stays (already landed, free).
  - **D3 ITCM code placement: DEFERRED TO HARDWARE, quantified.** Hot-code
    working set = 310 lines (9.9 KB) fits 64 KB ITCM outright. Steady-state
    throughput win ≈ 0 (I-cache resident; AN4891 flash+cache = ×1.01). The
    real win is **determinism**: it deletes the I-side cold/jitter WCET term
    (≈ 310 × 72 ≈ 22 K cycles ≈ 5.6% of the 1 ms Daisy callback budget).
    Needs a libDaisy linker section + startup copy (BSP territory) and a
    board to verify — the top on-target item, not actionable here.
  - **D4 F2 osc-vs-IFFT embedded pricing: IFFT justified for dense frames.**
    Osc-bank cost per dense partial per 256-sample block = 256 × 21.3 ≈
    5.45 K cycles [validated]. A Q31 inverse RFFT-512 is structurally
    22–31 K instructions (radix-4: 4 stages × 64 butterflies × ~70–90 insns
    + split + bitrev) → 15–60 K cycles across the IPC band [structural
    bound, not a measurement] → **crossover at ~3–12 simultaneous dense
    partials** even at the pessimistic end. Typical dense workloads exceed
    this comfortably ⇒ the F stream is justified; its FIRST step is
    vendoring CMSIS-DSP and measuring `arm_rfft_q31` under the counts rig
    (rig ready; maintainer's CMSIS-DSP emphasis lands exactly here). Until
    then the oscillator bank remains the only signed-off path.
- **Deterministic capacity (Daisy Seed, STM32H750 @400 MHz, 48 kHz, chosen
  SDRAM timing, DTCM placement contract).** Two number classes — never mix:
  **WCET-bound (guaranteed)** from `m7-wcet` (every term an upper bound) and
  **expected steady-state (modeled, validated ≤1% on loops)**:
  | scenario | WCET-bound | expected steady-state |
  |---|---|---|
  | 48-sample callback (Daisy default, 1 ms, 400 K cyc) | **32 voices @ 88%** (24 @ 71%, ~36 @ 100%) | ~128 voices ≈ 43% load (28 cyc/vs whole-kernel) |
  | 256-sample blocks (2.13 M cyc) | **128 voices @ 89%** (≈145 @ 100%) | ~290 voices compute ceiling; `DAISY_MAX_ACTIVE`=128 binds first |
  Segments: storage-bound at **2,000,000** (`DAISY_MAX_SEGMENTS_SAFE`; 48 MB
  SDRAM pool, 16 B each; raw pool fits 3.1 M); per-block scan ≈ 48 cyc/check
  bound, streamed-scan 14–25 cyc/segment (P4). Headroom for non-audio work =
  100% − the table's load (e.g. ≥11% guaranteed at the 128-voice/256-block
  point, ~50%+ at 64 voices). 96 kHz halves every budget; 480 MHz boost adds
  20%. The WCET bound at the 48-sample callback is dominated by deliberately
  conservative terms (the 256-block-measured residual charged whole on a
  48-sample block + cold memory per block); on-target measurement will pull
  the guaranteed numbers toward the expected column, never past it.
- **Real newlib + working sets (hardening pass, 2026-06-10):** the freestanding stub
  headers are deleted; the rigs require a newlib toolchain (sha-pinned xPack via
  `m7-bootstrap` into `tools/toolchains/`, gitignored; brew's bare gcc is rejected by
  a libc probe). Measured before adopting: DSP census identical stub-vs-newlib; with
  `-ffreestanding` kept the full mnemonic census is byte-identical. The runner links
  `-lc_nano` (newlib-nano memcpy/memset — the family Daisy firmware links) and pins
  FPSCR=0 at reset per the fidelity contract. The counts plugin now also measures
  unique 32B-line working sets per range and per region (footprints for P4) and has
  an optional full address-trace mode (`trace=` arg) for offline cache simulation.
  New baseline on xPack 15.2.1: `spectral_arm32_process` = 3,762,455 insns (was
  3,763,544 on brew 15.2.0 — compiler scheduling delta), output checksum 77a267f6
  unchanged, counts bit-reproducible; per-block kernel data working set ≈ 260 lines
  (~8.3 KB) — comfortably DTCM-resident, a measured input for P4/P6.
