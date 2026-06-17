# Arch decoupling & path-selection — which mechanism, when

This codebase selects divergent code paths (host vs embedded, vDSP vs portable, CMSIS vs
scalar, AVX2 vs SSE, CPU vs Metal/CUDA) using **five** different mechanisms. They are not five
arbitrary idioms for one job — each fits a *different kind* of fork. This doc states the rule so
the choice is predictable, and records why the parts that look inconsistent are actually
principled. It complements AI_CANON's "capability-not-CPU gating" rule.

## The spine is sound — do not churn it

`core/` owns the **contract headers** every port implements — `spectral_io.h` (out-kernels),
`spectral_synth_ifft.h` (iFFT backend), `spectral_oscillator_dispatch.h` (SIMD segment),
`spectral_synth_internal.h` (`gpu_tile_preprocess`, the shared `SegmentLoopParams` POD),
`spectral_vector_ops.h`. `arch/{arm,simd,ref}` and `drivers/{vdsp,metal,cuda}` hold the per-port
bodies. This is textbook port-contract design. No arch body reaches into private core state:
`SegmentLoopParams` is a clean POD read only through documented fields. **Leave the spine alone.**
The only taxonomy blemish is cosmetic (below); the arm/arch decoupling concern, while reasonable,
does not point at a real coupling defect.

## The decision rule

Pick the mechanism by **what the fork actually is**, not by habit:

| The fork is… | Mechanism | Examples |
|---|---|---|
| **1. Two mutually-exclusive port bodies** fulfilling one contract header, chosen by a **profile decision CMake owns** (host vs embedded) | **New file + CMake file-selection.** The body has **zero** profile `#if`. | `out_kernels`, `osc_simd`/`cmsis`, `vector_ops`, `gpu_tile` — `SPECTRAL_SOURCES_*_HOST` vs `_EMBEDDED` in `source-manifest.cmake:55-94` |
| **2. Two mutually-exclusive port bodies** gated by a **compile-time capability the platform resolves** (not CMake), where **both TUs must co-link** in one test/parity binary | **Whole-file self-`#if`** (one body compiles to nothing). | iFFT: `arch/ref/spectral_ifft_ref.c` `#if !SPECTRAL_USE_VDSP` + `drivers/vdsp/spectral_ifft_vdsp.c` `#if SPECTRAL_USE_VDSP` |
| **3. Orthogonal hardware-capability flags** stacking within **one ISA's** TU (DMA? DTCM? dual-MAC? CMSIS-DSP?) — they co-occur, they are not alternatives | **In-body `#if` on a named capability predicate.** | `arch/arm/spectral_synth_arm32.c` (`SPECTRAL_HAS_DMA`, `SPECTRAL_USE_CMSIS`, DTCM placement, …) |
| **4. Width / lane parametricity** (same algorithm at 4- and 8-wide) | **Re-includable width-templated `.inc`.** | `arch/simd/spectral_oscillator_simd_kernel.inc`, `spectral_fast_sin_simd.inc` (`OSC_VW=4|8`) |
| **5. Runtime user-selectable backend** (CPU / Metal / CUDA / export) | **Function-pointer vtable chosen at init.** | `core/spectral_backend.c` `SpectralBackendVTable`; `spectral_oscillator` per-timbre dispatch |

## Why case 2 (iFFT self-`#if`) is NOT an inconsistency to "fix"

It looks like it should match case 1 (out_kernels' CMake file-selection), but three verified facts
make case 1 **impossible and wrong** here — converting it would be a regression:

1. **`SPECTRAL_USE_VDSP` is a C-preprocessor fact, not a CMake variable.** It is derived from a
   platform check in `spectral_config.h:621-625` (`#ifndef` → 1 on Apple, else 0). CMake never
   sets it. A `if(SPECTRAL_USE_VDSP)` file-select would have to **re-derive the `__APPLE__`→vDSP
   mapping in CMake**, creating a second source of truth — exactly the C-truth violation the
   doctrine forbids (no CMake copies of C facts).
2. **There is a documented escape hatch:** a build may force the portable path with
   `-DSPECTRAL_USE_VDSP=0` (`config.h:618-620`), and tests use it
   (`core-contracts-test.cmake:18`, `peak-interp-parabolic-test.cmake:17`). A CMake-side file
   choice keyed on `APPLE` would **desync** from a `-D` override — link the vDSP TU while the C
   body believes it is portable. The self-guard makes the compiled body == the body `config.h`
   selected, always.
3. **The self-guard's real job is duplicate-symbol avoidance so both TUs co-link.** Both define
   the same four symbols (`spectral_ifft_backend_create/destroy/n_fft/inverse`), and
   `ifft-synth-parity-test.cmake:14-15` lists **both** unconditionally in one executable
   (`spectral_ifft_vdsp.c` hard-includes `<Accelerate/Accelerate.h>`, absent off-Apple). CMake
   file-selection cannot host two same-symbol bodies in one link. The whole-file `#if` lets one
   portable source list compile cleanly on Apple **and** Linux.

So out_kernels uses file-selection because it is a profile decision CMake owns; iFFT uses a
self-`#if` because it is a platform-resolved capability that must co-link. **Different mechanisms
because different requirements** — the rule above predicts both correctly.

## Hard constraints the rule encodes

- **The m7 perf gate freezes `arch/arm/spectral_synth_arm32.c` byte-for-byte.** Its dense in-body
  `#if` (case 3) is load-bearing AND untouchable: splitting its orthogonal capability axes into
  files is both meaningless (they co-occur) and codegen-fatal (adding any out-of-line function
  renumbers GCC `.L` labels → forces a baseline regen). Case 3 is *correct*, not a smell.
- **Parity tests** (`ifft_synth_parity`, `gpu_backend_parity`, `full_fused_parity`, `osc_parity`)
  often need two ports co-resident in one binary → case 2.
- **The embedded build is libc-free / no-malloc.** Case 5 (vtable + a registry) fits host/cold
  backend selection only; it is disqualified on the embedded hot path (indirect calls defeat
  per-sample inlining and move the pinned codegen).

## The one cosmetic blemish (low priority)

`arch/ref/` conflates two roles: genuine portable-reference **algorithms** (`spectral_ifft_ref.c`,
a working radix-2 iFFT) and a 12-line presence/absence **stub** (`spectral_gpu_tile.c` returning
`SPECTRAL_ERR_BACKEND_UNAVAIL`). The gpu_tile split is a *desktop-has-GPU vs everyone-else-stub*
**capability** gate, not an ISA port — `source-manifest.cmake:82-90` already annotates this. A
reader expects `arch/ref/` = "the portable fallback algorithm," not "the GPU-absent no-op." Fix is
documentary or a relocation of the stub; pure taxonomy hygiene, not worth production-wiring churn.

## Bottom line

The architecture is ~80% already consistent with this rule. The maintainer's unease is real but
the diagnosis "we use macros where we should use files" is mostly **not** the situation: the five
mechanisms map to five genuinely-different fork types. The high-value, near-zero-risk move is to
**make the rule predictable (this doc)** — not to migrate code. Every concrete migration the design
study proposed (iFFT→CMake, a unified capability header, the gpu_tile move, a `RESTRICTED_*` rename)
was shown to be either wrong-on-the-facts or churn-for-cosmetics on a perf-pinned, parity-gated tree.
