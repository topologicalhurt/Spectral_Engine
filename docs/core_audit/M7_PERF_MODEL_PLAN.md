# M7 Performance Model — Campaign 3 / S1 measurement stack

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

## Hot kernels under the model

`synth_core_m7` (4×-unrolled coupled-step + MAC), `synth_core_pair_m7` (SMLALD dual-voice),
`synth_fade_m7` (fade ramp), `spectral_coupled_step` (4×SMULL/SMLAL class per sample, inlined
into all three), plus the block-level scan/activation/prune in `spectral_arm32_process` and
the segment-load/DMA path.

## Phases

- **P0 — survey + decision.** DONE (this doc is the record).
- **P1 — codegen rig committed.** `tools/perf_model/`: stub headers + one script that
  compiles the real TUs, extracts the hot-loop bodies, emits the instruction census.
  Done-when: one command regenerates the census for the four hot kernels.
- **P2 — QEMU counts rig.** Freestanding runner + TCG plugin + driver script; deterministic
  fixtures shared with `arm_core_test` so correctness and counts come from the same inputs.
  Done-when: real `spectral_arm32_process` runs under qemu-system-arm and per-kernel
  instruction/memory counts are reproducible exactly.
- **P3 — pipeline layer.** Automated mca over extracted loop bodies + the validation
  microbench set with TRM cross-check. Done-when: cycles/iteration per kernel with stated
  validation delta.
- **P4 — memory layer.** Line-footprint analyzer over L1 traces; AN4891-calibrated latency
  table; SDRAM segment-stream/DMA model. Done-when: per-block stall estimate with per-constant
  calibration provenance.
- **P5 — integration.** Re-derive or retire `spectral_perf_model.c` profiles from the stack
  (decision point: the sim earns its keep here or is deprecated); WCET report; feeds S4
  benchmark redesign.
- **P6 — application.** Re-evaluate S1 optimization candidates against the model: loop-nest /
  data-layout inversion, SMLALD coverage beyond full-sustain pairs, ITCM code placement,
  and the F2 oscillator-vs-IFFT embedded crossover.

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
  `python -m spectral_tools.testing.benchmark_workflow m7-census | m7-counts`
  (CMake: `m7_census` / `m7_counts`, same family as `bench`). The workload fixture is
  a Python SSOT that generates the C header the runner compiles; reports carry the
  fixture digest and a measured/modeled provenance tag on every number. Harness
  behavior is itself under test: `pytest tests/tools` (extraction/parsers/fixture
  units + skip-aware end-to-end census and reproducible-counts integration; the
  Daisy↔toolchain flag pairing is asserted against options.cmake). Migration
  verified: kernel-range counts and mca per-loop cycles bit/numerically identical
  to the pre-migration rig.
