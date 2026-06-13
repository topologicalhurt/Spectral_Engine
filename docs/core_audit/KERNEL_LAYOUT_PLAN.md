# Kernel layout refactor (S3) — Linux-kernel-style separation

Maintainer mandate (2026-06-13): restructure the engine on the Linux kernel's
model — the kernel proper is architecture-independent; `arch/` holds
ISA-contingent implementations; `drivers/` holds device/library-specific
backends; boards live outside. Survey basis: 6-lens fleet map (pass 259) over
synth/, backends/, core/port/, build wiring, sim usage, GPU coupling — the
violations cited below are file:line-verified by that survey.

## Target tree

```
spectral_engine/
  core/        the kernel: arch-independent algorithms, contracts, types,
               dispatch. Gains: synth/math/spectral_q15.{h,c} (the Q-domain
               map — five core headers already include it upward), the synth
               public API header (synth/api/spectral_synth.h, future
               spectral_kernel.h 1.0), the CPU synthesis driver
               (backends/cpu — portable, OpenMP = kernel service), the IFFT
               frame renderer + contract (synth/spectral_synth_ifft.*).
  arch/        ISA-contingent kernels, build-selected per target, capability
               macros mapped ONCE in spectral_config.h (the Kconfig analog):
    ref/       portable C fallbacks — always compilable, the host-oracle
               substrate (current port/embedded stubs + portable branches +
               the radix-2 iFFT port).
    arm/       ARMv7E-M: spectral_synth_arm32.*, SoA layout, DSP paths, and
               the sim TU as its HOST ADAPTER (see Q2 below).
    simd/      host-SIMD via SIMDe (port/host oscillator_simd, vector ops,
               out kernels, gpu_tile host) — ISA-portable SIMD, not x86.
  drivers/     device/library backends behind core contracts:
    metal/     .m TU + the MSL payload TU (moved OUT of core/oscillator.c)
               + the committed generated header + codegen wiring.
    cuda/      .cu TU (already a true SSOT consumer of core headers).
    vdsp/      Apple Accelerate ports (spectral_ifft_vdsp.c now; the
               analysis vDSP branch later).
    cmsis/     CMSIS-DSP ports (F4 Q31 iFFT).
  runtime/ analysis/ cmd/   unchanged this campaign (kernel services,
               subsystem, userland).
api/daisy_seed/             board support, already correctly outside.
```

## The six maintainer questions, answered on survey evidence

**Q1 — backends into core?** Yes for `cpu` (portable synthesis driver =
kernel code). No for `arm` (ISA-contingent → `arch/arm/`). `cuda`/`metal`
move to `drivers/` — mechanically clean (two thin TUs exporting 9 functions),
but four anchors must be cut first (Phase L4): (a) the MSL payload is
`#include`d by core/oscillator.c:295, compiling shader strings into every
target as dead rodata and forcing verify_metal_osc onto all six targets —
move the payload to a drivers/metal TU, gate shrinks to desktop; (b) the
GPU extern declarations live in the public API header behind `__APPLE__`
sniffing — move to per-driver headers included by the dispatch TU only;
(c) the MSL struct mirrors (SegmentGpu/SynthParams/TileRange as strings,
"no compile-time check possible") get codegen'd from the C structs like the
oscillator formulas already are; (d) the vendor-NEUTRAL plan/cache layer in
core/spectral_synth_internal.c stays in core — that part is already correct
(the DMA-helper analog). spectral_q15.{h,c} → core (Q1 confirmed: the
include graph already treats it as core; synth/math placement is the lie).

**Q2 — does sim serve a purpose?** Yes, narrowly, and the survey proves it:
it is the ONLY end-to-end host run of the full pipeline over the real
embedded kernel (WAV → analysis → float→Q15 conversion/normalization policy
→ real spectral_arm32_process → WAV), and the float→Q15 conversion policy is
exercised nowhere else (arm_core_test covers the kernel, the QEMU rig covers
counts — neither covers the conversion). Verdict: KEEP as `arch/arm/`'s host
adapter, after surgery: extract segment_to_q15 into core (it duplicates
cmd/convert_segments.c by its own admission), move report printing to the
caller (a backend calling console printers is a layering bug), delete the
dead API (embedded_sim_set_config/verbose — zero callers) and the dead
define (SPECTRAL_SIMULATION_TARGET_DAISY — zero consumers).

**Q3 — rewiring needed?** Found by the survey, queued as Phase L0 (small,
independent): (1) BUG: `measure --target simulate` resolves the DESKTOP
binary (glob `spectral_*_desktop` can never match
`spectral_*_native_simulation`) — the simulate row of the measurement matrix
has been silently wrong; (2) cmd/cli/main.c calls metal_cleanup/cuda_cleanup
directly instead of the vtable; (3) the GPU timbre cap is defined three
times; (4) the api header reaches into backends/arm for the synth_cpu macro
redirect; (5) core/spectral_synth_internal.c calls synth_cpu by symbol —
route the fallback through the CPU vtable entry.

**Q4/Q5 — synth/ and core/port/?** Both dissolve. synth/api → core;
synth/math → core; backends/cpu → core; backends/arm + sim → arch/arm;
backends/gpu → drivers/; the IFFT contract+renderer → core with ports in
drivers/vdsp + arch/ref. core/port/host → arch/simd (SIMDe TUs) and
drivers/vdsp (library TUs); core/port/embedded → arch/ref (stubs) or
arch/arm (DSP bodies); port/spectral_mem.h (the placement capability
contract) → core. After L5 neither `synth/` nor `core/port/` exists.

**Q6 — the framework.** Four rules, enforced not aspirational:
1. **Contract headers in core** (include/linux analog): one header per
   subsystem surface; drivers/arch implement, never extend, the contract.
2. **Build-system selection** (Kconfig/arch-Makefile analog): the source
   manifest lists per target select exactly one implementation TU per
   contract; capability macros live in the single arch→capability map in
   spectral_config.h. No #ifdef mono-files for anything bigger than an
   inline primitive (pass-256 ruling).
3. **Include direction is law**: arch/ and drivers/ include core/; core
   includes neither; cmd bypasses no vtable. Enforced by a structural test
   (tests/tools/test_layering.py) that parses the include graph and fails on
   any upward edge — a dependency-DAG contract, not source-text theater.
4. **Every port pair gets a parity test** (window_backend_parity pattern;
   the GPU paths currently have ZERO automated coverage — the survey's
   sharpest finding — and get a parity test when they move).

## Phases (each lands green: ctest + pytest + the m7-baseline perf gate —
file moves must be codegen-identical, and the gate PROVES it at ±5% insns)

- **L0 — survey bug fixes** (independent): the simulate-binary glob bug,
  dead sim API/define removal, vtable cleanup in main.c, timbre-cap single
  constant.
- **L1 — Q-domain home**: synth/math/spectral_q15.{h,c} → core/; update the
  five core includers, manifest, 7 test cmakes, daisy relative include,
  toolchain.py KERNEL_TU_RELPATHS + ENGINE_INCLUDE_SUBDIRS (rig paths break
  loudly otherwise — by design).
- **L2 — arch/ stand-up**: create arch/{ref,arm,simd}; move port/host SIMDe
  TUs + port/embedded TUs + backends/arm + sim (post-surgery from Q2);
  retire core/port/.
- **L3 — synth/ dissolution**: api header + cpu backend + IFFT files → core;
  fix the api→backend redirect inversion; synth/ removed.
- **L4 — drivers/ stand-up**: gpu → drivers/{metal,cuda} with the four
  anchor cuts; ifft vdsp port → drivers/vdsp; MSL mirror codegen; GPU parity
  test added.
- **L5 — enforcement**: include-graph layering test; ENGINE_INCLUDE_DIRS
  narrowed per layer; docs (AI.md tree description) trued up.

## Status

- Survey complete (fleet map), plan authored, maintainer answers delivered.
- **L0 DONE** (split across the Master Review + one commit here): simulate
  glob bug + dead sim API/define (review W2), timbre-cap constant (W4),
  main.c GPU-cleanup vtable bypass → spectral_backend_cleanup_all().
- **L1 DONE**: spectral_q15.{h,c} → core/; manifest + 8 test cmakes + daisy
  relative include + toolchain.py rewired; synth/math/ gone; gate-proven
  codegen-identical (ctest 24, pytest 75, gate PASS).
- L2 next: create arch/{ref,arm,simd}; move port/host SIMDe TUs +
  port/embedded TUs + backends/arm + sim (after the Q2 surgery: extract
  conversion policy, move report printing to the caller); retire core/port/.
