# Embedded performance & determinism surface

The single, canonical surface for measuring and gating the embedded (Cortex-M7 /
Daisy Seed / Teensy 4.1) synthesis path. One Python entry point, one report shape,
every number tagged **measured** vs **modeled**.

> **Entry point:** `python -m spectral_tools.testing.benchmark_workflow <command>`
> (add `PYTHONPATH=tools`). Every command emits the same `{suite, context,
> tests:[{name,status,summary,details}]}` report through `core/report_output.py`
> (pretty tree by default; `--raw` for JSON; `--json-out PATH` to write it).

## The one question this answers

**Determinism (device D) :=** the modeled worst-case execution time (WCET) of one
audio block at the canonical workload (512 voices) is ≤ the real-time budget of D,
with the safety margin, under D's worst-case memory/cache state. Determinism is
**always tethered to a device** (clock, cache, TCM, external-memory timing) — the
Daisy STM32H750 @ 480 MHz today; Teensy i.MX RT1062 planned. See
`docs/core_audit/active/DETERMINISM_SURFACE_PLAN.md` for the full design.

```
PYTHONPATH=tools python -m spectral_tools.testing.benchmark_workflow deterministic
```
→ **PASS/FAIL** for "512 voices @ 480 MHz", the deterministic ceiling, and the
t1/t2/t3 WCET breakdown. (Today: FAIL @ ~154% of budget on the exact oscillator;
ceiling ~277 voices — 512 needs the IFFT or fade-pairing+TCM-residence.)

## How to generate each metric

| You want… | Command | Provenance |
|---|---|---|
| **Determinism PASS/FAIL @ target** | `deterministic [--voices N] [--block B]` | modeled (WCET) |
| How code compiles → instructions (per loop) | `m7-census` | measured (codegen) |
| Instruction counts + **bandwidth** (ld/st bytes, 32B lines) | `m7-counts` | measured (QEMU TCG) |
| Modeled **cycles** per hot loop + validation | `m7-census`, `m7-mca-validate` | modeled (llvm-mca CortexM7Model), validated vs TRM |
| **IPC** (sustain kernel) | `deterministic` (`execution_time.ipc_sustain_kernel`) | modeled |
| **Stalls / memory bounds** (bandwidth↔serial range) | `m7-stalls` | modeled (P4, over a measured trace) |
| **WCET / ACET / BCET** | `deterministic` (`execution_time`), `m7-wcet --active N` | modeled+measured |
| Regression gate vs frozen baseline | `m7-baseline` (`--generate` to re-sign) | measured+modeled |
| **Hottest functions/lines** | `hotspots [--voices N]` | measured (PC-histogram → addr2line) |
| **FPU/MAC/ALU/SIMD saturation** | `saturation` | **modeled** (llvm-mca port pressure / cycles; M7 has NO PMU) |
| Idle / interrupts / on-device cycles | *planned (P3 on-device DWT contract)* | measured (DWT, erratum-aware) |
| Matrix of target × instrument | `measure --list` / `measure --target m7` | — |

## What is and isn't sound (read this before trusting a number)

Grounded in the WCET literature + ARM docs (`reference/` + the plan's research section):

- **QEMU TCG is functional, not cycle-accurate.** It gives *exact instruction and
  byte counts*, never cycles. Cycles come from llvm-mca's `CortexM7Model` (a static
  pipeline model), validated against hand-derived TRM timings (`m7-mca-validate`,
  ≤1% body delta). The QEMU+llvm-mca split is the same hybrid MCAD uses (<3% error),
  though MCAD itself doesn't model the M7.
- **WCET here is a conservative *hybrid bound*, not a formally-sound aiT proof.**
  Sound WCET needs abstract-interpretation + IPET (per-arch, state-explosion); the
  M7 is the documented hard case (random-replacement caches + dual-issue; ~102% MAE
  for tight analysis). We instead bound each layer at its worst extreme (cold
  all-miss cache, CPI bound, one mispredict per data-dependent branch) + a named
  safety margin. **Measurement alone underestimates WCET — never gate determinism on
  a measured max.**
- **The determinism worst case is param-invariant** (proven: `test_determinism_invariance.py`).
  The embedded path is sine-additive, so timbre/frequency/pitch/stretch/amplitude
  reduce to "N sine voices" and don't change the per-voice kernel cost; only voice
  count + fade-state do, both bounded by the 512-all-fade case.
- **DWT erratum 850724** (M7 r0p1/r0p2/r1p0): the on-device profiling counters
  mis-attribute (LSU stalls → CPICNT; FPU lazy-stacking → LSUCNT). On-device reports
  must record the CPUID revision and treat those counters as combined/suspect.
- **TCM placement is the M7 determinism lever** — ITCM/DTCM are single-cycle (no miss
  variability); the cache stance is "TCM-resident hot path + cold-miss bound the rest."

## Layers (M7_PERF_MODEL_PLAN P0–P6)

```
L0 codegen census  codegen.py            how source compiles to instructions
L1 dynamic counts  counts.py + spectral_counts.c (QEMU TCG plugin)   insns/bytes/lines
L2 cycle model     codegen.py (llvm-mca) + mca_validation.py         modeled cycles, validated
L3 stall bounds    memory_model.py       cold-cache/SDRAM serial+bandwidth bounds
   WCET            wcet.py               t1 synth + t2 residual/scan + t3 cold-memory
   gate            expectations.py       frozen baseline + stone ceilings + determinism gate
   device SSOT      api/<device>/...sdram.h (C-truth; Python parses, holds no copy)
```

Regenerating the baseline (`m7-baseline --generate`) is a deliberate re-signing of
the performance contract — only for an intended change, stated in the commit.
