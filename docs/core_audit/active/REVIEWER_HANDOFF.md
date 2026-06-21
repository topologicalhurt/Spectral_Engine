# Spectral Kernel — Reviewer Handoff Mandate (Campaign 3)

**STATUS: RECONCILED 2026-06-21 — standing doctrine, not a live ticket queue.** Every concrete
Campaign-3 work-stream is DONE or correctly GATED: S1 (perf model P0–P6, archived
`M7_PERF_MODEL_PLAN`), S2 (adversarial fleet ×2 + the hardening campaign, archived), S3 (refactor
bulk landed; STFT-unify/stub-collapse DECLINED-in-writing), S4 (embedded benchmarks + the tests_all
race fix), and F (the IFFT crossover measured ≈7 partials, the path implemented + parity-tested, and
the renderer abstraction generalized it — see `RENDERER_ABSTRACTION_PLAN.md` + AI_CANON #31). The two
items that were the live frontier are now DONE: the **`spectral_kernel.h` public-API surface** LANDED
at **v0.0.1** (commits 70b5bd6 + d61f559 — the umbrella header + the `kernel_api_freeze` guard pinning
version/layouts/enum-values/signatures; the hard 1.0 freeze is deferred, the library isn't 1.0-ready
per S5), and **S5** is DELIVERED (the scorecard at §S5 below). The live frontier is now the maintainer's
prioritized **maturity push: (1) the real-time embedded device, (2) GPU — harden + port more DSP onto
it, (3) the reproducible reference kernel, (4) the C library (at 0.0.1)** — plus the **F3 golden** (IFFT
default-on). The S5 punch list enumerates the per-axis gaps (host-doable vs hardware/CI-gated).
The §0 doctrine still stands; **`PLAN_CLOSURE_LEDGER.md` is the live status of record** and supersedes
the §6 doc-map below (whose `CAMPAIGN_2`/`OPTIMISATION`/`OSCILLATOR_BACKEND_CONTRACT` targets are now
in `archive/` + `reference/`).

> You are an advanced AI coding agent taking over a mature-but-unfinished DSP kernel. This is
> your standing mandate, not a ticket. It is deliberately **abstract in places**: the maintainer
> wants your architectural and problem-solving judgment, not literal box-checking. Where this
> doc says "decide", "investigate", or "use discretion", that is real authority — exercise it,
> justify it with evidence, and record the decision. There is **no deadline**; this runs over a
> long horizon. Optimize for the *most correct, most mature* outcome, not the fastest one.

---

## 0. How to operate (the non-negotiable contract)

The previous campaigns established a discipline. Inherit it strictly — it is why this codebase
is trustworthy:

1. **Measure first, assert never.** No optimization, no "this is faster", no architectural claim
   ships without evidence. Characterize *before* implementing (build the experiment, run it, read
   the number). When the data says an idea loses, **decline it and document why** — declining on
   data is a success, not a failure. (This session rejected the magic-circle oscillator and an
   amp-in-Q15 SIMD path on measured data; both were correct calls.)
2. **No theater.** Code must not pretend to be more performant/correct than it is. Comments that
   claim an optimization must be backed by it or deleted. Inert attributes, dead "fast paths",
   fictitious linker sections, uncalibrated cost models = bugs. (This campaign removed exactly
   these from the ARM path.)
3. **Verification anchors** (keep them green; extend them):
   - `ctest` (19 tests as of pass 250; was 14 when this mandate was written) — the regression
     contract. Build `tests_all` then `ctest`. See `REVIEWER_HANDOFF_2.md` for what has landed
     since this mandate and the current open frontier.
   - `arm32_process_correctness` — the host-runnable **real M7 codepath** oracle (property-based:
     frequency / amplitude / finiteness over a sweep). It forces `SPECTRAL_ARM_M7=1` and
     `SPECTRAL_HAS_DUAL_MAC=1` so the optimized path is exercised on the host via portable
     fallbacks. **This is your ground truth for any ARM change.**
   - `arm-none-eabi-gcc -mcpu=cortex-m7 …` codegen inspection — prove the instructions you intend
     (SMLALD, etc.) are actually emitted. (Bare toolchain here lacks newlib; use `-ffreestanding`
     or minimal TUs, as prior work did.)
   - `AI_CANON.md` — the rules (exact-by-default, approximations only behind a gate + parity test,
     SSOT for constants, no doc/test load-bearing on source text, capability-not-CPU gating).
4. **Capabilities, not CPUs.** Core kernels gate on capability flags (`SPECTRAL_HAS_DUAL_MAC`,
   …), and a single arch→capability mapping lives in `spectral_config.h`. The embedded surface
   will grow to **all low-level ARM A/M profiles first, then RISC-V** — design every new fast path
   so a new arch extends the mapping, not the kernels. Optimize the bulk target first (Linux model).
5. **Don't assume a hosted libc on embedded.** `<string.h>` is not freestanding; memory ops route
   through `spectral_mem_zero` (libc-free). Apply the same discipline to anything new.
6. **Orchestrate when it pays.** You have workflows / sub-agent fleets. Use them for fan-out
   (multi-perspective audits, doc surveys, migrations) and **adversarially verify findings**
   (3-skeptic majority-refute caught ~48/49 false "findings" this session — most plausible
   findings are wrong; verify before acting). But do the cheap thing inline.
7. **Commit discipline.** Terse per-commit notes (AI_CANON §18). Commit only working, verified
   units. The maintainer pushes; you commit locally on the working branch unless told otherwise.

---

## 1. The project in one screen

Spectral analysis → resynthesis kernel in C. Pipeline:
`WAV → STFT (window + FFT) → spectral-peak pick/estimate → peak tracking → sinusoidal Segments → additive synth → audio`.

Two performance-critical worlds, one codebase:
- **Desktop/host**: float synthesis, Metal/CUDA GPU, vDSP/FFTW, SIMDe→AVX, OpenMP, large (>256 MB)
  STFT datasets, a block-chain allocator.
- **Embedded ARM Cortex-M7** (Daisy Seed / STM32H750): Q15/Q31 fixed point, DSP extension
  (SMLAD/SMLALD/QADD16/SSAT), FPv5-D16 **double** FPU, DTCM(128 KB, zero-wait)/ITCM/AXI-SRAM/
  SDRAM(64 MB) hierarchy, I/D cache + DMA.

Layout: `spectral_engine/{core,analysis,synth,runtime,cmd}`, `api/daisy_seed`, `tests/`,
`docs/core_audit/`. Build: `cmake -B build` then targets `desktop`, `simulate`/`simulate_daisy`,
`embedded_arm[_float|_restricted]`, `cuda`, `daisy`, and `tests_all`. Read `AI.md` (root) for the
tight orientation and `spectral_engine/README.md` for the feature map.

---

## 2. State of play (reconciled against the real code — Campaign 2 + the ARM/embedded campaign closed)

**Done / trustworthy now:**
- Correct, host-verified ARM synth (the real M7 path runs on host under a green oracle).
- Exact **q63 accumulator** (proper mix headroom, saturate-once).
- **M7 dual-MAC (SMLALD)** wired via full-sustain voice-pairing (codegen-confirmed).
- Memory placement **bound to the real Daisy BSP sections** (`.dtcmram_bss`/`.sdram_bss`) via
  `SPECTRAL_BSP_MEM_HEADER` — no more fictitious sections (the old "DTCM placement" was inert).
- `SPECTRAL_HAS_DUAL_MAC` capability flag; libc-free `spectral_mem_zero`; DMA cache-invalidate
  derived from buffer placement (a latent coherency bug, fixed).
- **Coupled-form Q31 oscillator** replaced the per-sample sine-LUT *gather* on the M7 path
  (deterministic latency, no cache miss, no table) — verified 82–140 dB SNR vs exact sin, oracle green.
- Cleanup campaign (122 grep-tests deleted, CI repointed at ctest, docs consolidated, AI_CANON
  rewritten, AI.md authored). Two intensive adversarial audits (perf; cache/bandwidth/OpenMP)
  concluded the host-level instruction/cache design is **sound** (the remaining wins are on-target
  or algorithmic).

**Known-open, by theme** (details in the active plans — see §6):
- **On-target / hardware-gated** (the frontier): real cycle/cache/DMA/WCET numbers; ITCM code
  placement (no ITCM section yet); LUT-residency now *moot* on M7 (no gather); perf-model
  calibration (the cost model is an uncalibrated heuristic).
- **Algorithm fork (F2)**: oscillator-bank vs inverse-FFT synthesis for dense frames — **not
  settled** (benchmarks not run; IFFT path not implemented). See the F-stream below.
- **Refactor debt**: stub "method" tokens (`hybrid_render`, `adaptive_track_density`,
  serra_smith/johnston no-op stubs), `full_matrix` vs fused STFT path, public-API shape
  (a future stable `spectral_kernel.h`), the C8 magic-number tail.
- **Verification debt**: golden-vector oracle sign-off (D2), LUT-generator feedback (D3),
  backfill of past defect-findings as compiled regression tests (D5), Macro-2 logical audit (L1–L6).
- **Feature gaps** (documented, mostly out of *core* scope): real-time file streaming, segment
  compression, the residual/psychoacoustic models (Serra-Smith / Johnston stubs), an on-device
  Daisy test suite.

**Maturity target = ALL FOUR at once** (the bar to audit against in §S5): a shipped real-time
**embedded device**, a reusable **C library with a frozen 1.0 API**, a reproducible
**research/reference kernel**, and a high-throughput **desktop/GPU performance product**. A change
that matures one dimension must not regress another (e.g. an embedded micro-opt that breaks the
library API, or a desktop SIMD path with no parity test).

---

## 3. The mandate — five work-streams + one algorithm fork

Sequencing is suggested in §5, but you hold discretion over order and decomposition. Each stream
names a **done-when** so you can tell genuine completion from motion.

### S1 — ARM/embedded performance down to the instruction & byte level
**Goal:** make the embedded path genuinely, *measurably* optimal — to the point where hand
assembly would be a wasted marginal effort — and prove it. Emphasis (the maintainer's words):
**cache coherence, memory bandwidth, DMA, saturating ALUs, and using the MAC / FPU / DSP units
when present.** Concretely:
- Exploit the hardware: dual-16-bit MACs (SMLAD/SMLALD — pairing currently only covers
  full-sustain voices; extend), saturating ALU ops (QADD/SSAT), the FPv5-**D16 double** FPU where
  double precision actually buys correctness cheaply (it already powers the per-block oscillator
  seed), and any DSP-extension op that removes instructions. Gate each on a **capability**, and
  verify the op is *actually emitted* (codegen) and *actually beneficial on that arch* before
  enabling.
- Memory hierarchy: pin hot data/code to DTCM/ITCM (the binding mechanism exists; ITCM needs a
  linker section), make the SDRAM segment stream DMA-double-buffered and **coherent** (the
  clean-before-TX / invalidate-after-RX contract), reason about cache lines and write-buffer
  saturation, and treat **WCET as a first-class output** (real-time has no average case).
- Consider the loop-nest / data-layout question the prior critique raised (sample-major /
  voice-parallel vs the current per-voice nest; SoA lane contiguity) — but **only if the model
  below says it wins**; an earlier analysis showed naive inversion just moves traffic.

**The measurement problem is part of S1 (and S4):** there is **no hardware**. Your deliverable is
**the most accurate model of Cortex-M7 performance obtainable without a board** — at the level of
*individual instruction cycles, cache/prefetch behavior, and bytes of memory moved*. The existing
sim (`synth/backends/sim` + `runtime/spectral_perf_model.*`) is crude and **may be deprecated
entirely if it earns its keep no other way**. Field-research the best approach and justify your
pick — candidates to evaluate (non-exhaustive, not endorsements): cycle-approximate emulation
(QEMU + timing, Renode), static per-instruction analysis (`llvm-mca` against an M7 scheduling
model), instruction-accurate ISS (OVPsim/Imperas-class), gem5, vendor cycle models, or a
**calibrated analytical model** built from the Cortex-M7 TRM instruction timings + the STM32H7
AXI/cache/SDRAM latencies. Whatever you choose: it must produce numbers you can defend, separate
**measured/assumption-free** quantities (instruction counts, bytes, codegen) from **modeled** ones,
and never present a heuristic as a measurement.

**Done-when:** the embedded hot paths are optimal against a *defensible* cycle+cache+bandwidth
model; every optimization is capability-gated, codegen-verified, and oracle-clean; the perf model
is calibrated (or honestly bounded) and its assumptions are explicit; WCET is reported.

### S2 — Adversarial bug-checker fleet (gated on an S1 milestone)
After S1 reaches a coherent milestone (the maintainer's "after a certain milestone" — you define
it; a reasonable bar is "the instruction-level ARM rework is structurally settled and green"),
launch a **large, extremely adversarial** correctness sweep across the *whole* kernel — not just
ARM. Use multi-perspective fleets (DSP, numerics, fixed-point/packing, GPU/SIMD/CMSIS, pipeline
end-to-end, build-flag matrix, concurrency/coherence) and **3-skeptic majority-refute** verification
(most findings are false; prove the real ones). Every confirmed lifecycle-spanning defect becomes a
**compiled regression test**, not a note. Fold in the open Macro-2 (L1–L6) audit and the D5 backfill.
**Done-when:** a verified defect list with fixes + permanent tests; the false-positive rate of the
fleet itself is reported (calibration of the audit).

### S3 — File-level refactor, dedup, wiring, architecture & design patterns
Pay off structural debt with an architect's eye for a **monolithic** project (the maintainer's
framing — this is a kernel, not a microservice mesh; choose patterns that suit a single tight
C codebase, not enterprise ceremony). Concretely: correct header/file separation and include
wiring; single-source-of-truth for logic (the prior dedup pass found duplicated formulas — keep
hunting), collapse the dead/stub "method" tokens into their real implementations or delete them,
unify or justify the dual STFT paths, and resolve the **public API** shape — a stable
`spectral_kernel.h` 1.0 surface is one of the four maturity targets, so design and freeze it with
SemVer discipline. Enforce the frameworks the codebase already declares (the contracts, the
capability model, the port-layer separation) rather than letting them erode. **Discretion is
explicit here:** decide how far to modularize vs keep monolithic, which patterns fit, and where
tech debt is worth paying vs leaving — justify each call.
**Done-when:** no dead/duplicated logic in the hot paths, wiring is correct and minimal, the public
API is frozen and documented, and the declared architecture is actually enforced (not aspirational).

### S4 — Performance & test benchmark redesign
Re-audit and **redesign the benchmarks** (perf and test) to be as accurate and robust as possible.
This is the twin of S1's measurement problem: the benchmarks must measure the *right* thing
(arithmetic intensity, bytes moved, cycles, cache pressure, WCET on embedded; throughput, tail
latency, bandwidth on desktop/GPU) and be **reproducible** (fixed inputs, stated variance, no
flakiness — the desktop `tests_all` parallel-build race is a known flake to fix). Tie embedded
benchmarks to the S1 model; tie desktop benchmarks to real perf counters where available. Make
"benchmark-then-decide" a first-class, documented lifecycle (DISCIPLINE_FINDINGS flags that not
all fast-path candidates have one).
**Done-when:** every fast path has an accurate, reproducible benchmark; the test suite covers the
lifecycle-spanning behaviors (not source-text grep); CI runs them.

### S5 — Final maturity audit (against all four targets)
After S1–S4, a final adversarial audit asking: **is the kernel mature** as (a) a shipped real-time
embedded device, (b) a 1.0 C library, (c) a reproducible reference kernel, and (d) a desktop/GPU
performance product? For each: what's missing for production (error handling, resource bounds,
API stability, determinism/WCET, parity/golden coverage, docs)? Produce a maturity scorecard with
the concrete gaps to 1.0 on each axis.
**Done-when:** a defensible per-axis maturity assessment + the remaining-to-1.0 punch list.

**S5 SCORECARD — DELIVERED 2026-06-21** (4-auditor adversarial audit, grounded in the tree at
ctest 40 + the `spectral_kernel.h` v0.0.1 surface). Verdict: **the algorithmic kernel is mature on
every axis; productization is the gap.** (Maintainer priority for the push: 1 embedded, 2 GPU, 3
reproducible kernel, 4 the C library — kept at 0.0.1, not 1.0.)

| Target | Grade | The core is strong… | …but to reach 1.0 |
|---|---|---|---|
| (a) Real-time **embedded device** (M7/Daisy) | **GAPS** | no-malloc static-pool deterministic Q15/Q31 synth; load-time validation of the untrusted `.spq`; a real qemu+llvm-mca+SDRAM perf gate vs a frozen baseline; SINAD≥70 dB on the real kernel; *(NEW 2026-06-21:* admission is now an explicit WCET knob `SPECTRAL_ARM32_ACTIVE_CAP` decoupled from storage + `arm32_admission_cap` ctest — commit 1d3b058)* | **BLOCKER:** nothing is validated on real silicon (every cycle/cache/DMA/WCET is host-modeled — QEMU TCG has no timing fidelity, DWT never read). ~~the active-voice cap isn't enforced at admission... a dense file overruns~~ **(CORRECTED:** the cap is now an explicit, bound knob — and the audit's "128 → overrun" premise was **stale**: 128 is a *legacy* figure, the modeled oscillator-bank ceiling is ~520 voices @400MHz / ~625 @480MHz [10000 cyc/sample ÷ ~16 cyc/voice-sample HIGHWORD-MUL kernel], peak speech polyphony is 176, and the 512 storage default is itself within budget — so there is no host-modeled overrun). The residual gap is that this WCET ceiling is **modeled, not silicon-validated**; no linked firmware ELF / Daisy test suite (libDaisy unvendored); the SD-load D-cache-invalidate fix is documented-not-wired. |
| (b) **C library** (v0.0.1, pre-release) | **GAPS** | a single curated `spectral_kernel.h` umbrella with semver + a compile-time surface guard; structured `SpectralError`; overflow-guarded allocation | **BLOCKER:** there is *no library target* — no `add_library`/`install`/`export`/pkg-config, so nothing external can actually link it. `analyze_audio` has no error channel (returns an empty `SegmentArray` on failure). Unprefixed public symbols (`analyze_audio`, `synth_cpu`) will collide in a host app. *(The version-story contradiction + the values-not-just-sizes freeze gap the audit flagged are now FIXED — commit d61f559.)* |
| (c) Reproducible **reference kernel** | **NEAR** | deep parity battery (full/fused, scalar-vs-SIMD bit-exact, q-ladder vs `__int128`, the new kernel-freeze + renderer tests); the §VIII firmware-faithful determinism fix; paper-backed methods | the `SPECTRAL_REPRO_BUILD` knob is non-functional when set directly + the "repro" profile still appends ThinLTO unconditionally; no *external* golden vector (every parity test is an internal A-vs-B cross-check); no test that the build is actually bit-reproducible; no `-ffile-prefix-map`/`SOURCE_DATE_EPOCH`. |
| (d) **Desktop/GPU** performance product | **GAPS** | mature SIMDe/OMP additive CPU path with bit-exact SIMD-vs-scalar parity + honest perf observability (wall/Busy/Idle, RSS, Faults); the perf frontier is measured + closed; *(NEW 2026-06-21:* `gpu_backend_parity` now covers **all** GPU-eligible timbres 0..parabola, not just sine/saw, **and** pins the ineligible-timbre→CPU fallback as bit-identical — commits b5d67d5, 0af9974)* | the GPU is a deliberate *approximation* (hardware `sin`/`asin`, not the CPU's documented formulas) — not a parity/deterministic backend; no evidence the GPU is *faster* (measured slower on the runnable workload); CUDA never compiled/parity-tested; Metal parity never runs in CI (virtualized runner skips it); GPU is additive-only (wavetable/subtractive/IFFT are CPU-only, and cubic phase isn't even packed into the 32-byte `SegmentGpu` — see the F3 GPU caveat in IFFT_SYNTHESIS_PLAN); no desktop/GPU throughput *regression* gate. |

**Remaining-to-1.0 punch list, by gate:**
- **Host-doable now** (no hardware/maintainer needed): add a real `add_library(spectral)` + `install`/`export` of the 8 public headers (the library-axis blocker); give `analyze_audio` an error channel (or an `_ex` twin); namespace the 4 unprefixed entry points with compat aliases; fix the `SPECTRAL_REPRO_BUILD` knob + make the repro profile actually LTO-free + add a build-twice reproducibility ctest + `-ffile-prefix-map`; ~~add a Metal-vs-CPU parity test over *all* GPU timbres~~ **(DONE — b5d67d5; + fallback contract 0af9974)** + a non-stationary fixture; add a desktop/GPU throughput regression gate; a consumer integration doc + public-API CHANGELOG.
- **Maintainer-gated:** the F3 golden sign-off (IFFT default-on); wiring the IFFT/wavetable hybrids into the production dispatch.
- **Hardware / CI-gated:** on-target M7 bring-up (link the firmware ELF vs libDaisy, flash a Daisy, capture DWT cycles, validate the WCET model + SD-load coherency + DMA path on silicon); silicon-validate the modeled voice-ceiling WCET @ block=48 (the admission cap is now an explicit knob — commit 1d3b058 — but its ceiling is host-modeled, ~520@400MHz/~625@480MHz, not DWT-measured; 128 was a stale legacy figure); a real-GPU CI runner for Metal parity; an x86 box to measure the AVX2/AVX-512 tiers + a CUDA GPU to verify the `.cu`.

### F — Algorithm fork: the synthesis-method decision (settle with data + discretion)
The dominant open Big-O decision (OPTIMISATION_PLAN F2): keep the per-partial **oscillator bank**,
or add an **inverse-FFT (Rodet-Depalle) synth** for dense-spectrum frames, or a hybrid switched by
partial density. The maintainer wants this **engaged, measure-first** — run the osc-vs-IFFT density
benchmark on desktop and on your S1 embedded model, settle the crossover, and decide/implement
with judgment (you may stage or defer implementation, but the *decision* should be data-backed, not
left dangling). Related: F1 (MQ track linkage — code infra exists, validation pending) and F3
(per-track cubic-phase interpolation). Behavior changes here are golden-signed-off.
**Done-when:** the synthesis-method question is answered with measured crossover data and a wired
decision (even if "oscillator-bank stays, here's the evidence").

---

## 4. Cross-cutting principles (read these as binding even where a stream is silent)

- **Discretion over literalism.** The five streams are the *shape* of the work, not its bounds. If
  the right move is something not enumerated here, do it and explain. Do not be myopic.
- **Correctness gates performance gates elegance.** Never trade down. A faster wrong kernel is a
  regression; an elegant slow one misses the point on embedded.
- **The four maturity dimensions are simultaneous constraints**, not a menu. Check a change against
  all four.
- **Long horizon, no rush.** Depth over speed. It is correct to spend a whole effort *characterizing*
  before touching code (S1's model is itself a project).
- **Leave the docs true.** When you finish a thread, update the active plan (or archive it) and the
  CHANGELOG so the next reader inherits ground truth, not stale status (this handoff exists because
  the docs had drifted from the code).

---

## 5. Suggested sequencing (non-binding — your judgment governs)

`S1 (incl. the perf model) → milestone → S2 fleet → S3 refactor → S4 benchmark redesign → S5 maturity audit`,
with **F (synthesis-method)** engaged opportunistically (its benchmark rides S1's model; its
decision should land before S5). S3 and S4 interleave naturally. Re-survey the docs and re-run a
small adversarial pass between major streams to catch drift.

---

## 6. Map to the surviving planning docs (open work lives here)

Active (keep current as you close items):
- `CAMPAIGN_2_MASTER_PLAN.md` — Campaign-2 master plan. **Status now stale in places** (Phase A2/A3 dual-MAC +
  DMA coherency + section binding LANDED this session; D4 CI + grep-test retirement DONE). Open:
  Phase D2/D3/D5 (golden oracle sign-off, LUT-gen feedback, regression backfill), and the on-target
  A2/A3/A4 frontier → **S1/S2/S4**.
- `OPTIMISATION_PLAN.md` — the algorithm/optimisation track. Open: **F2 synthesis method** (→ F),
  O2/O3/O5 micro-ops (gated), O5-B stub collapse (→ S3).
- `OSCILLATOR_BACKEND_CONTRACT_PLAN.md` — backend matrix; open LUT-scale (32700 vs 32767) decision
  and CMSIS-into-embedded-dispatch wiring (→ S1/S3).
- `QTYPE_REFACTOR_PLAN.md` — Q-type threads; remaining tail is x86-CI / AVX-512 gated (→ S1/S4).
- `DISCIPLINE_FINDINGS.md` — the multidisciplinary requirement list (public API, benchmark
  lifecycle, C8 tail) → **S3/S4/S5**.

Reference (inherit, don't re-derive): `AI_CANON.md`, `CORE_CONTRACTS.md`,
`FULL_FUSED_PARITY_HARNESS.md`, `KERNEL_PATCHING_GUIDELINES.md`, `ACADEMIC_SOURCES.md`,
`CHANGELOG.md`, `VALIDATION_OWNERSHIP.md`. Completed campaigns are in `archive/`.

Top-level: `AI.md` (orientation), `spectral_engine/README.md` + `TODOs.md` (feature map + gaps),
`api/README.md`.
