# Patch notes — Pass 142: make the M7 codepath host-buildable (ARM verification enabler)

## Problem

The real embedded synth (`spectral_arm32_process` / `synth_core_m7`) only ran on
Cortex-M hardware; on the host the simulate build used a *separate reimplementation*
(`synth_arm32_simulation`). To make the sim a faithful oracle / perf model over the
REAL code (ULTRAPLAN A1b), the actual M7 codepath must be buildable and runnable on
the host. Three things blocked that.

## Change

```text
1. SPECTRAL_ARM_M7 is now overridable (#ifndef guard) so a host build can force the
   M7 codepath with -DSPECTRAL_ARM_M7=1. Default detection unchanged.
2. spectral_data_sync_barrier(): the dsb instruction is now emitted only on real
   Cortex-M (defined(__ARM_ARCH_7EM__|7M__)); on host (incl. forced-M7 host-sim) it
   uses __atomic_thread_fence(SEQ_CST) (ordering only). dsb alone does not assemble
   on x86 / needs an operand on arm64.
3. TCM/SDRAM section attributes (SPECTRAL_DTCM/ITCM/SDRAM) are gated on real ARM in
   addition to M7+EMBEDDED, so a forced-M7 host build does not emit
   __attribute__((section(".dtcm_data"))) (invalid on macOS Mach-O).
```

## Verification

```text
- spectral_synth_arm32.c compiles on host with -DSPECTRAL_ARM_M7=1 (0 errors;
  spectral_arm32_process present, synth_core_m7 inlined in). The Q15 math uses the
  portable intrinsic fallbacks in spectral_q15.h, bit-identical to the DSP path.
- No regression: `make simulate` green and the interim oracle's 6 cases still match
  (existing host/embedded builds are unaffected — all three changes are gated on the
  real-ARM arch macros or the new override).
```

Real-ARM cross builds (embedded_arm*) preserve prior behavior (the dsb and the TCM
sections are gated on the same `__ARM_ARCH_7EM__/7M__` that previously enabled them)
but were not built here — no ARM toolchain in this environment; verify on-target.

## Follow-up

Pass 143 wires the CTest harness target (`tests/arm_core`) that links and runs the
real `spectral_arm32_init/load/process` on the host (forced-M7), asserting audio —
the correctness half of the A1b split. The sim then becomes the perf/resource model
over the same real code.
