# Oscillator Backend Contract & Q15 Unification Plan

> **Status:** design captured 2026-06-06; maintainer chose "unify all Q15 first".
> **Executed 2026-06-06 (PASS222):** Phase 0 (matrix in program design + docs) ✅,
> Phase 1a (divergence characterized) ✅, Phase 1d (contract versioned) ✅.
> **Phase 1b DECLINED-on-evidence** (the physical LUT merge + the arm32→canonical reroute
> both change shipped embedded output or damage a deliberate harness design — surfaced to
> maintainer below). **Phase 1c DEFERRED** behind that maintainer decision (a true bit-parity
> gate is only meaningful after the LUT-scale question is resolved).
> **Executed 2026-06-06 (PASS223):** Phase 2a (CMSIS-Q15 oscillator kernel on the canonical
> contract — 3rd version-pinned consumer; `arm_sin_q15` declined on anti-drift grounds;
> backend-uniform dispatch declaration; matrix doc) ✅. **Phase 2a-WIRING SURFACED** to maintainer
> (no live caller — promoting CMSIS-Q15 into the embedded dispatch reverses the PASS216
> `#if !SPECTRAL_EMBEDDED` Q15 guard + changes shipped embedded output). **Phase 2b** is
> hardware-gated and does not even compile-check locally (no `arm_math.h`/libDaisy); strongest
> local check = `clang -fsyntax-only` against an `arm_math.h` shim (exit 0).
> **Executed 2026-06-06 (PASS224):** Phase 3 (autonomous part) ✅ — `osc_backend_contract`
> source-scan CTest makes the Q15 anti-drift matrix SELF-ENFORCING: every file that includes
> the canonical `spectral_osc_q15.h` (discovered by glob, not hard-coded) MUST carry a
> `_Static_assert(SPECTRAL_OSC_Q15_VERSION == …)` pin, so a future Q15 backend that forgets to
> handshake the contract fails CI. Two-part by design: the COMPILER owns pin *value* (a bump
> breaks `== 1`), this scan owns pin *presence* (which the compiler is blind to). The plan's
> "fold the parity CTests into ONE monolith" was **declined on data** — the six modular parity
> gates each pin a distinct cell with its own budget; collapsing them loses diagnosability for
> zero coverage gain. The new gate is added ALONGSIDE them (13th test). Float gets no per-includer
> pin rule (its formulas are a shared `static inline` source of truth; its re-implementers — SIMDe,
> GPU-Metal, CUDA — are gated by runtime parity / `verify_metal_osc` codegen, stronger than a token).
> **Phase 3 remainder is hardware-gated, not a local gap:** GPU-Metal *runtime* parity (needs an
> Apple GPU) + CMSIS *runtime* Q15 parity (PASS223 hardware-deferred).
>
> **Executed 2026-06-06 (PASS225):** Phase 4 (vDSP math-accel audit) ✅ — measured every production host
> vector op in `spectral_vector_ops.c` against its vDSP/vForce equivalent under production host flags
> (`docs/core_audit/VDSP_MATH_ACCEL_AUDIT.md`, harness `tests/core_contracts/bench_vdsp_audit.c`, not in
> CMake). **ONE genuine high-value win: vForce `vvatan2f` for the phase path** — 3.3×–11.5× over the EXACT
> scalar `atan2f` loop (production runs exact, `SPECTRAL_ENABLE_APPROX_ATAN2=0`) at ~1 ULP; it is the only
> compute-bound op and feeds the per-frame STFT phase path (`spectral_analysis_fft.c:365`). **Surfaced as a
> maintainer decision, NOT wired** (moves default-desktop output ~1 ULP → breaks byte-identity; adds an
> Accelerate dep to the core host path; host-only so scalar fallback stays). DECLINED on data:
> vmul/vadd/vsq/vsmul (bandwidth-bound, lose at shipping STFT sizes), `magsq_only` (fused SIMDe beats 3-pass
> vDSP). Marginal/optional: `vmax`/`vmaxmgv`, `magsq_split`.
>
> **Executed 2026-06-06 (PASS226):** Phase 5 (GPU Q15 double-pack audit) ✅ — **MEASURED WIN**, the
> maintainer's instinct vindicated. On M1 Pro the GPU oscillator's hardware fp16 `sin` runs ~2.5× the fp32
> rate; the half2 double-pack (two output samples/thread, fp32 phase + accumulate retained) is **2.0–2.7×**
> faster on the faithful synth inner loop at a **~−67 dBFS** precision floor — the OPPOSITE of CPU B2
> (half/Q15 slower on NEON). **GO-candidate, surfaced as a maintainer decision, NOT wired:** land OPT-IN
> behind a quality flag (not a default flip — GPU is fp32-exact today, ~−67 dBFS is not ≤1 ULP).
> `docs/core_audit/GPU_Q15_DOUBLEPACK_AUDIT.md`, harness `tests/core_contracts/bench_metal_q15pack.m`,
> CHANGELOG Pass 226. **This closes the proposed Phase 0→5 order.** Remaining open cells (CMSIS-Q15 live
> wiring, GPU-Metal/CMSIS runtime parity, LUT-scale convergence) are hardware-/maintainer-gated.
> All landed work is additive/byte-identical on every locally-buildable target: the 5 standard
> product builds are green (desktop, simulate, simulate_daisy, embedded_arm, embedded_arm_float);
> ctest 13/13 green.
> (The non-standard `embedded_arm_restricted` target is independently broken —
> `_analyze_audio`/`_perf_print` undefined — confirmed pre-existing by stashing these edits and
> reproducing the identical link error; unrelated to oscillator work.)
> **Relationship to prior plans:** [OSCILLATOR_UNIFICATION_PLAN.md](OSCILLATOR_UNIFICATION_PLAN.md)
> unified the *float* oscillator onto one L1 contract (`spectral_osc_formulas.h`), and
> [QTYPE_DOMAIN_PLAN.md](QTYPE_DOMAIN_PLAN.md) / [QTYPE_REFACTOR_PLAN.md](QTYPE_REFACTOR_PLAN.md)
> shipped the opt-in Q15 compute kernels and a starter discipline layer. This plan closes the
> remaining gap: make **every oscillator backend** (Scalar / SIMDe / CMSIS / GPU) implement **one
> versioned contract per compute domain** so they cannot drift, and **unify the three Q15 worlds**
> onto the canonical Q15 contract before any new backend work lands.
> **North star (unchanged):** float is the DEFAULT compute domain; Q15's primary home is embedded;
> desktop Q15 is opt-in (`--q15`); the default desktop render stays byte-identical when Q15 is not
> opted in.

---

## 0. Maintainer intent (2026-06-06, verbatim)

Original directive:

> "We should also implement the oscillator in vDSP, Cmsis in addition to SimDE and for maximum
> maintainability/robustness enforce a design contract and pattern so that they don't drift. When all
> is said and done we should have an ossilliator for: GPU, SimDE, vDSP, CMSIS, Scalar. The Q15 path
> only needs to propogate into SimDE, Scalar and CMSIS but we should consider if it is worth
> implementing for vDSP and GPU as well. We need to make it clear in the documentation and the program
> design which paths prohibit q15 (if any.) Furthermore, we also need to document what paths an embedded
> build can take. I expect SimDE, CMSIS and Scalar only to work on embedded targets (flag this if thats
> unexpected) which means they all have to support q15 (every embedded path must be able to support
> purely q15 or a q15 + float math that would be designed to maximally saturate the FPU and the integer
> unit)"

Three forks resolved (maintainer answers, verbatim):

1. **vDSP** — "Ok correction: we should look for any and all oppurtunities to use vDSP to accelerate
   performance on math but don't require a vDSP osccilator." → **No vDSP oscillator backend**; instead a
   measure-first audit of where vDSP/Accelerate can accelerate engine math.
2. **Embedded Q15** — "Unify all Q15 first." → Converge the three Q15 implementations onto one contract
   **before** adding backends.
3. **vDSP/GPU Q15** — "Unless it makes sense to double pack on GPU (it should? But I'm unsure) we can keep
   it float only." → GPU/vDSP stay float-only by default; **investigate GPU Q15 double-packing**
   (measure-first), promote only on a proven win.

---

## 1. Grounded current state (verified 2026-06-06, with file:line — not recited)

**The backend set is already 4-of-5 built. vDSP is the only "missing" one, and per fork (1) it is not
an oscillator.**

- **Scalar** — `core/spectral_osc_formulas.h` (canonical float L0, `SPECTRAL_OSC_FORMULAS_VERSION` +
  `_Static_assert` guards) consumed by `core/spectral_oscillator.c`. Universal (host + embedded).
- **SIMDe** — `core/port/host/oscillator_simd.c` (+ `oscillator_simd_kernel.inc`,
  `oscillator_simd_scalar_waves.h`). The **desktop default CPU path**, and the embedded-*simulator* path
  (itself a host x86 build).
- **CMSIS** — `core/port/embedded/oscillator_simd.c`: a **complete CMSIS-DSP float oscillator**
  (`arm_sin_f32`/`arm_mult_f32`/`arm_add_f32`/`arm_scale_f32` for sine/saw/triangle/parabola; scalar
  fallback for square/asin/quantized/pwm — `osc_simd_available()` advertises the first four). Gated
  `#if defined(OSC_SIMD_CMSIS)`, which `spectral_oscillator_dispatch.h:8` defines **only** for
  `ARM_MATH_CM4/CM7/ARMV8MML`.
- **GPU** — Metal MSL (codegen'd from the C contract) + CUDA `.cu`. Float.
- **Dispatch contract already exists** — `spectral_oscillator_dispatch.h`: per-timbre 2-bit `OscDispatchWord`
  (lines 33–45), the `osc_simd_segment_*` interface every CPU-SIMD backend implements (73–79), and the
  Q15 pack8 entry `osc_simd_q15_segment` (84–92, `#if OSC_SIMD_GENERIC` — i.e. **SIMDe/host only**).

**Premise correction (maintainer flagged to confirm).** The expectation "SimDE, CMSIS and Scalar only
work on embedded" is inverted versus the tree:

| Backend | Runs on | Embedded-hardware? |
|---------|---------|--------------------|
| **Scalar** | host **and** embedded | yes (universal fallback) |
| **SIMDe** | host only (desktop default + embedded *sim*, which is a host build) | **no** — real Cortex-M swaps it for CMSIS at `spectral_oscillator_dispatch.h:8` |
| **CMSIS** | embedded Cortex-M only | yes (the only embedded-exclusive path) |
| **GPU (Metal/CUDA)** | host | no |
| **vDSP** | host (Apple) — **analysis/FFT only today**, no oscillator | no |

So the actual **embedded-hardware oscillator paths are Scalar + CMSIS(float) + the arm32 Q15 synth** —
SIMDe never executes on real Cortex-M. The "every embedded path must support Q15" requirement therefore
lands on **CMSIS + arm32**, not SIMDe. (All 5 green targets are `SPECTRAL_EMBEDDED_SIMULATION=1`,
`CMakeLists.txt:130`; real CMSIS compiles only on the Daisy firmware, `daisy-config.cmake:87`.)

**The Q15 landscape — three worlds, one already-canonical contract.**

- **Canonical Q15 contract EXISTS**: `core/spectral_osc_q15.h` — "the single source of those evaluators,
  shared by production (`spectral_oscillator.c`) and the precision/parity CTests (no drift between what we measure
  and what we ship)." It carries the lone float→Q boundary (`spectral_osc_q15_phase_from_rads:40`), the
  gain-matched LUT (`spectral_osc_q15_init_sine_lut:53`), and pure Q15 evaluators for
  sine/saw/square/triangle/parabola inside a `SPECTRAL_Q_DOMAIN` region (68–92). Phase is the integer-NCO
  (`core/spectral_phase_nco.h`, `core/spectral_phase_nco8.h`).
- **Consumers of the canonical contract today**: `core/spectral_oscillator.c` (scalar Q15) and
  `core/port/host/oscillator_simd.c` (the pack8 8×Q15 SIMDe kernel).
- **The divergence** (the unification target): `synth/backends/arm/spectral_synth_arm32.c` includes
  `spectral_lut.h` + `arm_math.h` but **not** `spectral_osc_q15.h` or the NCO headers — it rolls its **own**
  Q15 oscillator. This is the one Q15 world not yet on the shared contract.
- **CMSIS oscillator has no Q15 variant** — it is float (`arm_*_f32`) only.

**Verdict.** Most of what the directive asks for is *contract enforcement and convergence*, not greenfield
backends: the float contract, the Q15 contract, the dispatch layer, and a parity-harness style (Phase D's
full/fused gate) all already exist. The genuine work is (a) pull arm32 onto the canonical Q15 contract,
(b) add a Q15 variant to CMSIS sharing that same contract, (c) a cross-backend parity gate so none drift,
(d) the vDSP math-accel audit, (e) the GPU Q15 double-pack investigation, plus (f) the documentation asks.

---

## 2. Phases (maintainer sequences them; proposed order + dependencies noted)

Every phase ends green on the standard gate: 5 builds clean + `ctest` all-pass + **default desktop render
byte-identical** when Q15 is not opted in. Measure-first / decline-on-data throughout.

### Phase 0 — Backend × domain × target matrix + documentation baseline (no code) — ✅ DONE (PASS222)
The two documentation asks, grounded in §1. Deliverable: an authoritative table of
{Scalar, SIMDe, CMSIS, GPU} × {float, Q15} × {desktop-host, embedded-sim, embedded-Cortex-M}, plus prose
for **"which paths prohibit Q15 and why"** (GPU/vDSP float-only pending Phase 5) and **"what oscillator
paths an embedded build can take"** (Scalar + CMSIS + arm32-Q15). Folds in the SIMDe-is-host correction so
the docs ship correct. Cheap; unblocks everything; pins reality before any refactor.
> **Landed:** the matrix is now an authoritative header-comment block at the top of
> `core/spectral_oscillator_dispatch.h` (the program-design locus for backend selection, per the
> maintainer's "documentation AND program design" ask) — backends × {float,Q15} × {host,embedded},
> "which paths prohibit Q15" (vDSP=not-an-oscillator, GPU=float-only, CMSIS-Q15=Phase 2), and the
> embedded oscillator paths (Scalar / CMSIS-float-today / arm32-Q15; SIMDe is host-only). vDSP is
> documented as math-accel-only (Phase 4), not a backend. No behavior change.

### Phase 1 — Q15 contract unification *(maintainer-chosen first)*
Converge the three Q15 worlds onto `spectral_osc_q15.h`.
- **1a — divergence characterization (no behavior change).** — ✅ DONE (PASS222). Findings:
  the "three Q15 worlds" are far *less* divergent than the planning framing assumed. For the SINE
  waveform all three already share the SAME interpolator `spectral_lut_sin` (osc_q15.h:72 wraps it;
  arm32 calls it directly at `spectral_synth_arm32.c` ~587). arm32 is **sine-only** — it implements
  no Q15 saw/square/triangle/parabola, so there is no non-sine divergence to reconcile. The only real
  divergences are: (i) **LUT amplitude scale** — embedded/arm32 builds via `spectral_lut_init_sine`
  at `SPECTRAL_LUT_AMP_SCALE = 32700` (`spectral_consts.h:38`), desktop/canonical builds via
  `spectral_osc_q15_init_sine_lut` at full-scale `Q15_MAX = 32767`; a uniform ~−0.0178 dB gain
  (20·log10(32700/32767)). This is **deliberate and already cross-documented** (osc_q15.h:47–52:
  full-scale gain-matches Q15 sine to float so parity reads as pure quantization, not gain). (ii)
  **Phase representation** — arm32 uses a 32-bit unsigned NCO accumulator (`phase >> 16` → uq16 index);
  the canonical sine takes a signed-Q15 `pq` and reinterprets `(uint16_t)pq`. Different conventions
  feeding the same primitive.
- **1b — unify.** — ⛔ **DECLINED-on-evidence (PASS222); surfaced to maintainer.** Neither sub-move
  is byte-identical or free:
    - *Collapse the two LUT builders to one shared algorithm:* there is no byte-identical extraction
      site. As a `static inline` in `spectral_lut.h` it pollutes a clean integer-only header (its
      current inlines never touch `sinf`/`SPECTRAL_TWO_PI`) across **every** includer, incl. embedded
      TUs. As a non-inline in `spectral_lut.c` it forces the two **deliberately zero-engine-link**
      precision harnesses (`q15_compute_precision`, `phase_nco_precision` — "header-only inlines plus
      the test TU, nothing linked from the engine", per their cmake comments) to link `spectral_lut.o`,
      breaking that design property. Cost (damage a deliberate design) exceeds benefit (de-dup ~10 lines
      of standard, already-cross-documented sine-table code). The drift risk is instead closed by 1d
      (version stamp) without touching either builder.
    - *Route arm32's sine through the canonical evaluator:* **not byte-identical** — it would change
      arm32's phase convention (32-bit NCO → signed-Q15) and its LUT scale (32700 → 32767), altering
      **shipped embedded output** (the `arm32_process` golden). Per the north star (don't gratuitously
      change embedded output), the numeric convergence — *should* embedded adopt full-scale 32767 and
      the canonical phase convention? — is a **maintainer decision**, not an autonomous refactor.
- **1c — anti-drift gate.** — ⏸ **DEFERRED behind the 1b maintainer decision.** A true *bit-parity*
  CTest across scalar-Q15 / pack8-SIMDe-Q15 / arm32-Q15 cannot pass today: the 32700-vs-32767 scale and
  the phase-convention gap are by design, so the test would have to bake in a known gain/phase offset
  (a *characterization*, not parity). The bit-exact cross-backend claim that IS provable today —
  scalar-Q15 ≡ pack8-SIMDe-Q15 — is already gated by `q15_simd_parity`. A clean bit-parity gate that
  folds in arm32 only becomes meaningful once the maintainer resolves the LUT-scale/phase question.
- **1d — version the contract.** — ✅ DONE (PASS222). `SPECTRAL_OSC_Q15_VERSION 1` added to
  `spectral_osc_q15.h` with a bump-rule comment (mirrors `SPECTRAL_OSC_FORMULAS_VERSION`), pinned via
  `_Static_assert` in the two production consumers: `spectral_oscillator.c` (scalar-Q15) and
  `core/port/host/oscillator_simd.c` (pack8 8×Q15, inside `OSC_SIMD_GENERIC`). A silent contract edit
  now fails their build until re-validated. Additive/byte-identical; ctest 12/12 green.

### Phase 2 — CMSIS Q15 oscillator (embedded SIMD)  *(depends on Phase 1)*
The "every embedded path supports Q15" goal. Today CMSIS is float-only.
- **2a — add Q15 evaluators to `core/port/embedded/oscillator_simd.c`** — ✅ DONE (PASS223).
  `osc_simd_q15_segment` + `osc_simd_q15_available` added under `OSC_SIMD_CMSIS`, consuming the canonical
  `spectral_osc_q15.h` (3rd `_Static_assert(SPECTRAL_OSC_Q15_VERSION==1)` consumer; the embedded sibling of
  spectral_oscillator.c `osc_q15_eval` and host `osc_q15_wave_scalar`, NOT a 4th copy). Structure: float quadratic
  phase + `spectral_osc_q15_phase_from_rads` boundary → canonical Q15 waveform (integer unit) → CMSIS-DSP
  `arm_q15_to_float`/`arm_mult_f32`/`arm_add_f32` widen+amp+accumulate (FPU) — literally the directive's
  "integer-unit Q15 eval concurrent with FPU float phase/amp." Declared backend-uniform in
  `spectral_oscillator_dispatch.h` (same signature as the SIMDe twin). **`arm_sin_q15` DECLINED on anti-drift grounds**:
  it is CMSIS-DSP's own Q15 sine table → a 2nd Q15 sine forked from canonical `spectral_lut_sin`, exactly the
  drift this initiative prevents. So CMSIS-Q15 is bit-parity with scalar/SIMDe-Q15 **by construction** (shared
  evaluator), already pinned by host `q15_simd_parity`/`q15_compute_precision`.
- **2a-WIRING — ⏸ SURFACED to maintainer (PASS223).** The kernel has **no live caller**: spectral_oscillator.c's whole
  Q15 dispatch is `#if !SPECTRAL_EMBEDDED` (PASS216), so embedded Q15 oscillator synthesis is owned by
  `spectral_synth_arm32.c` today and `g_osc_q15_sine_lut` is host-only. Promoting CMSIS-Q15 into the live
  embedded dispatch (a) reverses that deliberate guard, (b) introduces a 2nd embedded Q15 backend alongside
  arm32 (which owns embedded Q15?), (c) changes shipped embedded output. Per the north star + maintainer-sets-
  the-order, that is a maintainer decision, not an autonomous refactor. Kernel is written/pinned/declared so
  the wiring is a small scoped change once decided.
- **2b — verification is hardware-gated** — ✅ as far as local reaches (PASS223). Refined: it does **not even
  compile-check** locally — no `arm_math.h`/libDaisy here, `DAISY_PATH` unset, so `make daisy` FATAL_ERRORs and
  the existing CMSIS *float* path is equally un-buildable (the whole CMSIS path is genuinely hardware-gated).
  Strongest local check done: `clang -fsyntax-only -Wall -Wextra -DARM_MATH_CM7` against a minimal `arm_math.h`
  **shim** (signatures only) → exit 0, zero diagnostics, version pin holds. Cycle/throughput (does the dual-issue
  pay on M7?) + on-hardware numeric run **deferred to hardware/QEMU**, like the A2/A3 ARM deferral. Flagged
  honestly, not asserted.

### Phase 3 — Backend contract hardening + unified parity matrix (anti-drift, all domains) *(depends on 1)*
Generalize "don't drift" beyond Q15. Output: adding a backend or editing the contract without updating the
others fails CI.

- **3a — pin-completeness gate.** — ✅ DONE (PASS224). `spectral_engine/cmake/scripts/osc_backend_contract.cmake`
  + `cmake/targets/osc-backend-contract-test.cmake` (ctest `osc_backend_contract`, pure `cmake -P`). Discovers
  (glob) every includer of canonical `spectral_osc_q15.h` and requires each to carry a
  `_Static_assert(SPECTRAL_OSC_Q15_VERSION == …)` pin; also asserts both contract headers still declare a
  version. This is the **self-enforcing** half: the compiler already guards pin *value* (a bump breaks `== 1`),
  the scan guards pin *presence* (a re-implementer that wrote no pin). Negative-tested: missing pin → FATAL,
  direct + named-constant pin forms → OK, missing `#define` version → FATAL.
- **"fold parity CTests into ONE monolith" — ⛔ DECLINED-on-data (PASS224).** osc_parity, osc_width_parity,
  q15_simd_parity, q15_production_parity, q15_compute_precision, full_fused_parity each pin a distinct cell with
  its own budget + failure signature; collapsing them loses per-cell diagnosability and churns six green tests for
  zero coverage gain. The matrix is already gated as a constellation; the only missing invariant (pin completeness)
  was added *alongside* them, not by rewrite.
- **Float = no per-includer pin rule, by design.** `spectral_osc_formulas.h` is a shared `static inline` source of
  truth that callers merely call (nothing to drift). Float re-implementers are gated elsewhere and intentionally
  NOT by the scan: SIMDe by osc_parity/osc_width_parity (runtime, stronger than a token), GPU-Metal by
  `verify_metal_osc` (MSL codegen'd from the C formulas; build fails on drift), CUDA likewise.
- **3b — runtime parity cells (hardware-gated remainder).** ⏸ NOT a local gap: GPU-Metal *runtime* oscillator
  parity needs an Apple GPU (today gated by codegen-verify only); CMSIS *runtime* Q15 parity is hardware-deferred
  (PASS223, parity-by-construction holds via shared evaluators pinned by the host q15 CTests). Both await hardware/CI.

### Phase 4 — vDSP math-acceleration audit (measure-first; NO oscillator) *(independent)* — ✅ DONE (PASS225)
Per fork (1). vDSP already does the FFT (`analysis/spectral_analysis_fft.c`) + windows
(`core/spectral_windows.c`). Audit other host math it could accelerate that it does not yet:
`core/port/host/spectral_vector_ops.c` (vadd/vscale/vmul → `vDSP_v*`), magsq/magnitude, the synth
accumulate/scale reduction, envelope ramps (`vDSP_vramp`), peak scans (`vDSP_maxv`). For each: prototype,
measure vs the current SIMDe path, **keep only measured wins**, decline-and-document the rest. Host/Apple
only.

> **Audit complete — full results + recommendation in `docs/core_audit/VDSP_MATH_ACCEL_AUDIT.md`;
> harness `tests/core_contracts/bench_vdsp_audit.c` (host/Apple, not in CMake — links Accelerate);
> patch notes PASS225. NO production code wired; promotion is a maintainer decision.**
> - **ONE genuine high-value win: vForce `vvatan2f` for the phase path** (`spectral_vatan2` + the
>   atan2 half of `spectral_magsq_phase`). 3.3×–11.5× faster than the EXACT scalar `atan2f` loop
>   (production runs exact — `SPECTRAL_ENABLE_APPROX_ATAN2 = 0`) at **~1 ULP** (max 2.4e-7). It is the
>   only compute-bound op; `magsq_phase` is the per-frame STFT phase path (`spectral_analysis_fft.c:365`).
> - **Surfaced as a maintainer decision, NOT wired:** promoting it (a) moves default-desktop analysis
>   output ~1 ULP (breaks byte-identity — crosses the north star), (b) adds an Accelerate dep to the core
>   host vector path, (c) is host/Apple-only so the scalar fallback stays. If accepted: `#if`-guarded vDSP
>   path + scalar fallback + a parity CTest budgeting ~1 ULP.
> - **DECLINED on data:** vmul/vadd/vsq/vsmul (bandwidth-bound; *lose* at shipping STFT sizes n=513/2049),
>   `magsq_only` (fused single-pass SIMDe beats the 3-pass `ctoz`+`zvmags`+`maxv`). **Marginal/optional:**
>   `vmax`/`vmaxmgv` (consistent but negligible absolute win, bit-identical), `magsq_split` (no caller).
> - Default desktop render byte-identical by construction (zero production source changed).

### Phase 5 — GPU Q15 double-pack investigation (measure-first; default float) *(independent)* — ✅ DONE (PASS226)
Per fork (3). Investigate packing two Q15 lanes per 32-bit GPU register (Metal `half`/packed types; CUDA
`__half2`) for the synth kernel. Characterize throughput **and** precision vs the float GPU kernel on
representative segment counts. Promote only on a proven throughput win at an acceptable precision floor;
otherwise **decline-and-document GPU as float-only**. (GPU is the one "float framework" where 16-bit
packing might genuinely pay — the maintainer's instinct — so it gets a real measurement, not an assumption.)

> **Audit complete — MEASURED WIN (the maintainer's instinct vindicated). Full results in
> `docs/core_audit/GPU_Q15_DOUBLEPACK_AUDIT.md`; harness `tests/core_contracts/bench_metal_q15pack.m`
> (host/Apple/Metal, not in CMake — JIT-compiles MSL via `newLibraryWithSource` like production); patch
> notes PASS226. NO production code wired; promotion is a maintainer decision.**
> - **PROVEN THROUGHPUT WIN, ~2–3× — the OPPOSITE of CPU B2 (half/Q15 slower on NEON).** On M1 Pro the
>   GPU oscillator uses the HARDWARE `sin` (`spectral_osc_metal_generated.h:33`), and fp16 `sin` runs
>   **~2.5×** the fp32 rate; the half2 double-pack (two output samples/thread) reaches **3.0×** pure and
>   **2.0–2.7×** on the faithful synth inner loop (fp32 phase + fp32 accumulate retained; win grows with
>   segment density). x86/NEON has no fp16-transcendental speedup → why CPU lost and GPU wins.
> - **GPU Q-island = CPU Q-island:** phase MUST stay fp32 (accumulates over segment), accumulation MUST
>   stay fp32 (sums many partials); only the innermost waveform `sin` narrows to `half2`.
> - **Precision floor ~−67 dBFS** (≈11-bit; fp16 10-bit mantissa) — WORSE than the CPU Q15 ~−84 dBFS
>   floor. So it is a quality-tradeoff mode, not a free win.
> - **GO-candidate, surfaced as a maintainer decision, NOT wired:** land it **OPT-IN** (analogous to
>   `--q15`) behind a quality flag with a ~−67 dBFS parity budget — NOT a default flip, because the GPU
>   path is fp32-exact today and ~−67 dBFS is **not ≤1 ULP** (so faster-path-should-default does not
>   auto-apply). Wiring = half2 variant of `synthesize_tile_parallel` + dispatch/quality flag + GPU
>   parity CTest + on-the-real-kernel re-measure (microbench is the ALU/SFU ceiling; real kernel adds
>   tiling/barriers/divergence half-sin can't accelerate → realistic win lower but positive).
> - **Matrix impact if promoted:** `spectral_oscillator_dispatch.h` gains a `GPU-Metal | Q15/half` (opt-in) cell —
>   the first non-float GPU cell — and Phase-0 docs update. Until then GPU stays documented float-only.

---

## 3. Cross-cutting invariants
- North star reaffirmed each phase: float default; Q15 home embedded; desktop Q15 opt-in; GPU/vDSP
  float-only unless a measured win (Phase 5) flips it.
- No phase may let one backend silently change estimator/window/fade policy — the parity gates (1c, 3)
  forbid it.
- Documentation (Phase 0) is updated by any phase that changes the matrix (e.g. Phase 2 adds CMSIS-Q15;
  Phase 5 may add GPU-Q15).
