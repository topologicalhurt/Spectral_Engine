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

## Complete census — every path-selection site, classified and verified

Every divergent-path site in `arch/`, `drivers/`, and the synth/analysis core, checked against the
rule. **Result: the tree is fully conformant** — every apparent exception is principled or a
recorded decision; no site needs a mechanism change.

**Case 1 — CMake file-selection (mutually-exclusive port body, profile decision CMake owns):**

| Surface | host/desktop TU | embedded/other TU | manifest selector |
|---|---|---|---|
| Oscillator SIMD | `arch/simd/spectral_oscillator_simd.c` | `arch/arm/spectral_oscillator_cmsis.c` | `OSC_SIMD_HOST` / `_EMBEDDED` |
| Out kernels | `arch/simd/spectral_out_kernels.c` | `arch/ref/spectral_out_kernels.c` | `OUT_KERNELS_HOST` / `_EMBEDDED` |
| GPU tile | `arch/simd/spectral_gpu_tile.c` (real) | `arch/ref/spectral_gpu_tile.c` (BACKEND_UNAVAIL stub) | `GPU_TILE_HOST` / `_EMBEDDED` |
| Synth backend | `core/spectral_synth_cpu.c` | `arch/arm/spectral_synth_arm32.c` (embedded), `…_simulation.c` (sim), `drivers/metal`, `drivers/cuda` | `SYNTH_CPU` / `_EMBEDDED` / `_SIMULATION` / `_METAL` / `_CUDA` |

Each selected body has **zero in-body *profile* `#if`**. Conformant. Two annotations: (a) `vector_ops`
has only a HOST TU (`arch/simd/spectral_vector_ops.c`); embedded takes the scalar fallback behind the
`spectral_vector_ops.h` interface guard — a future bare-metal counterpart would land in `arch/ref/`
(manifest comments). (b) GPU tile is a *capability* split (desktop-has-GPU vs stub), not an ISA port —
see the cosmetic blemish above; the manifest already annotates it.

**Case 2 — whole-file self-`#if` (platform capability that must co-link):**

| File | guard |
|---|---|
| `arch/ref/spectral_ifft_ref.c` | `#if !SPECTRAL_USE_VDSP` |
| `drivers/vdsp/spectral_ifft_vdsp.c` | `#if SPECTRAL_USE_VDSP` |

The canonical, verified-principled case (see above). Conformant. Note: `arch/arm/spectral_synth_arm32.c`
and `spectral_synth_simulation.c` ALSO carry a whole-file `#if SPECTRAL_EMBEDDED` / sim guard, but those
are **belt-and-suspenders on top of CMake selection** (the file is already CMake-selected for
embedded/sim; the guard makes it a defensive no-op if pulled into the wrong link) — intentional, not a
pure case-2.

**Case 3 — in-body capability `#if` (orthogonal hardware flags on one ISA):**

| File | `#…` directives | axes |
|---|---|---|
| `arch/arm/spectral_synth_arm32.c` | 124 | EMBEDDED / M7 / CMSIS / DMA / DTCM / RESTRICTED — **frozen by the m7 baseline** |
| `arch/arm/spectral_debug_embedded_arm.c` | 23 | embedded debug instrumentation |
| `arch/simd/spectral_vector_ops.c` | 24 | `__AVX2__` width (hand-written 256+128) |
| `arch/simd/spectral_oscillator_simd.c` | 16 | `__AVX2__` width tier (drives the `.inc`) + Q15 |
| `arch/ref/spectral_out_kernels.c` | 11 | CMSIS / M7 (the embedded out-kernel) |

Orthogonal capability axes correctly in-body. Conformant.

**Case 4 — width-templated re-includable `.inc`:**

| `.inc` | instantiated by |
|---|---|
| `arch/simd/spectral_oscillator_simd_kernel.inc` | `spectral_oscillator_simd.c` at `OSC_VW = OSC_KERNEL_W (4|8)` |
| `arch/simd/spectral_fast_sin_simd.inc` | the kernel `.inc` (and `core/spectral_synth_ifft.c` at 4-wide) |

Conformant. **Recorded exception:** `vector_ops.c` hand-writes the 128+256 widths with in-body
`#ifdef __AVX2__` rather than a `.inc`. This is a *deliberately declined* unification (the un-256'd ops
were found to be dead code; recorded in `OPTIMISATION_PLAN`/QTYPE work), not a defect — case-4-via-in-body
is acceptable when the `.inc` extraction was measured not worth it.

**Case 5 — runtime vtable (user-selectable backend):**

| Surface | vtable |
|---|---|
| Synth backend (CPU / Metal / CUDA / Export) | `core/spectral_backend.h` `SpectralBackendVTable`, chosen at runtime by `spectral_backend_vtable()` |
| Oscillator timbre / quality | `spectral_oscillator` dispatch |

Conformant — cold/host backend selection, correctly **not** extended to the embedded hot path.

## Resolved items (closure)

- **gpu_tile-as-capability blemish** → DECIDED: document-only. The manifest already annotates the
  host-capability nature; relocating the stub is churn for cosmetics. No action.
- **`vector_ops` hand-written widths vs the `.inc`** → DECIDED: a recorded measure-first decline, not a
  defect. No action.
- **arm32 / simulation belt-and-suspenders `#if SPECTRAL_EMBEDDED`** → intentional defensive double-guard.
  No action.
- **iFFT self-`#if`** → verified principled (not a CMake-selectable profile decision; co-link required).
  No action.

No genuine inconsistency was found that warrants a code change. The census is complete.

## Bottom line

The architecture is fully consistent with this rule once it is *stated*: the five mechanisms map to five
genuinely-different fork types, and every site conforms. The maintainer's unease pointed at
*unpredictability* (no written rule), not at wrong mechanism choice. The durable rule now lives in
**AI_CANON #20**; this document is the complete verified census + the iFFT-exception rationale behind it.
Every concrete migration the design study proposed (iFFT→CMake, a unified capability header, the gpu_tile
move, a `RESTRICTED_*` rename) was shown to be wrong-on-the-facts or churn-for-cosmetics on a perf-pinned,
parity-gated tree. **Status: COMPLETE — archived.**
