# Spectral Kernel Ultraplan — Campaign 2

This is the master plan for the second core campaign. Campaign 1 (the host-kernel
audit, passes 1–137) is closed per `MASTER_PLAN_CLOSURE_CRITERIA.md`; its two
explicitly-deferred items were "ARM/embedded redesign as a separate project" and
"compiled full/fused parity harness". This campaign delivers those plus the
contract/guarantee, adversarial-correctness, and tooling-feedback workstreams.

## Governance (applies to every pass)

```text
- One focused change per commit, recorded as a terse line in docs/core_audit/CHANGELOG.md.
- Every change must justify itself as one of: real bug, real ownership cleanup,
  real API-surface reduction, real test/harness improvement, real perf-path
  simplification (KERNEL_PATCHING_GUIDELINES.md §7).
- Correctness before performance (AI_CANON §13). A faster wrong kernel is worse
  than a slower reference.
- When a contract changes, update AI_CANON.md / CORE_CONTRACTS.md /
  ACADEMIC_SOURCES.md / DISCIPLINE_FINDINGS.md as affected (AI_CANON §18).
- Validate at boundaries via spectral_contracts.h; no alias wrappers (§19).
```

## Delivery order (maintainer-directed)

```text
Phase A  ARM / embedded redesign            (immediate item: "arm code")
Phase B  Contract / guarantee registry      (immediate item: "WOLA/COLA contract")
Phase C  CTF / KISS adversarial sweep       (later item: "KISS pass")
Phase D  Compiled harness + tooling feedback (immediate item: "testing"; Phase G)
Phase E  Core port-layer separation          (later item: "refactor arches into files")
```

"Remove ARM NEON / deprecated code" was folded into Phase A (done, passes
140-141). The other "later" structural item — "refactor arches into separate
files" — turned out larger than first scoped: SIMD portability is a *capability*
concern already handled by SIMDe, but the real mess is embedded (low-level /
fixed-point) and host (float) implementations interleaved by #ifdef inside ~22
core/analysis files, plus device-specific detail leaking into core. That is now
its own stage, Phase E. The LUT generator feedback loop is folded into Phase D,
because the golden-vector oracle it depends on is built there.

### Sequencing risk and mitigation

The compiled harness (Phase D) is the natural numerical oracle for the ARM
redesign (A) and the CTF sweep (C), but the maintainer has scheduled it last.
Mitigation:

```text
- Phase A uses the EXISTING ARM simulation parity path as its interim oracle
  (synth_arm32_simulation, hardened in passes 132/137) plus per-pass golden
  output vectors captured before each refactor.
- Phase C findings are fixed immediately with the fix documented in the pass
  note; formal compiled regression tests are backfilled in Phase D (D5).
- Escalation: if A's numeric churn outruns the interim oracle, pull D0+D2
  (CTest infra + golden-vector fixtures) forward ahead of the rest of D.
```

---

## Phase A — ARM / embedded redesign

Objective: replace nominal, comment-asserted "optimizations" with a design that
actually exploits the Cortex-M7 (Daisy / STM32H7: DSP extension, fpv5-d16 FPU,
DTCM/ITCM/AXI-SRAM/SDRAM hierarchy), with explicit cache-coherency and
memory-bandwidth contracts. Files: `synth/backends/arm/*`, `synth/math/spectral_q15.*`,
`core/oscillator_simd.c`, `core/spectral_vector_ops.*`, `core/spectral_config.h`.

### A0 — Structural reset (folds in the two "later" items)

```text
- Establish a per-arch file layout under synth/backends/ so arch code is
  backend-specific only (ARCHITECTURE_CLEANUP_STATUS principle). Move shared
  scalar reference math out of the arch files.
- Remove the __ARM_NEON path in spectral_q15.c: Cortex-M (the embedded target)
  has no NEON; that block only ever compiled on desktop Apple-Silicon and is dead
  for the real target. Desktop SIMD is SIMDe's job (oscillator_simd.c).
- Delete other deprecated/contrived scaffolding flagged during the sweep.
- Capture golden output vectors for representative fixtures BEFORE touching the
  hot loop, to anchor the interim oracle.
```

### A1 — Correctness defects (fix before optimizing, AI_CANON §13)

```text
- Dual LUT-reader divergence: the 4x-unrolled body calls spectral_osc_lookup()
  while the scalar tail and fade call spectral_lut_sin() (arm32.c:596-599 vs
  611/636). Prove sample-identical or unify; otherwise the 4-sample boundary is
  a discontinuity. (At SPECTRAL_OPT_LEVEL>=2, spectral_osc_lookup is
  nearest-neighbour — quantization noise relative to the tail.)
- Frequency-increment units contract: freq_inc = freq_q88 * freq_inc_scale_q24
  (arm32.c:821, scale at :481) evaluates to omega*2^32/sr, where a turns/sample
  accumulator needs (omega/2*pi)*2^32. Derive the units against the float->Q15
  converter; encode as a contract test (AI_CANON §9). Confirm or fix.
- spectral_q15.h:16-18 declares uq32_t three times (copy-paste; intended uq8_t /
  uq16_t). Clean up the fixed-point type set.
- Audit Q15/Q31 saturation on worst-case overlapping-segment accumulation
  (DISCIPLINE_FINDINGS, hardware perspective).
```

### A1b — ARM verification foundation (oracle + sim rework)

The sim (`synth_arm32_simulation`) was a parallel reimplementation, not the real
`spectral_arm32_process`, so the interim hash oracle never exercised the redesign
target. Decided rework (must precede A2/A3):

```text
1. Host-run the real M7 codepath: a build mode forcing SPECTRAL_ARM_M7 on with
   EMBEDDED=0, DMA=0, so spectral_arm32_process/synth_core_m7 run on the host via
   the portable intrinsic fallbacks (spectral_q15.h) and no-op placement/prefetch.
   Only the dsb barrier needs a host-safe fence. Q15 math is bit-identical to HW.
2. CTest correctness harness (tests/arm_core): drive the REAL
   spectral_arm32_init/load/process over fixtures; assert audio vs golden and/or
   desktop-float reference within tolerance. Wires in spectral_arm32_load +
   validate_segment_data (currently orphaned).
3. Sim = perf/resource MODEL over the SAME real code. Separate MEASURED
   (assumption-free: op counts, bytes moved, cache-line touches, active counts,
   memory high-water) from MODELED (cycles/WCET, cache misses via explicit,
   single-sourced, QEMU/hardware-calibratable cost params). Artefacts: memory
   bandwidth, cache-miss model, cycles/WCET, memory high-water + DTCM/SDRAM.
4. Delete the parallel synth_arm32_simulation functional loop (one impl only).
   Retire tests/arm_oracle/oracle.py (interim) once the CTest harness lands.
```

Closure (A1b): real spectral_arm32_process runs on host; CTest asserts its audio;
sim reports calibratable perf artefacts over the same code; no duplicate ARM synth.

### A1b status (passes 142–158) — COMPLETE

```text
All four A1b rework items are done; the closure condition holds.

- Item 1 (host-run the real M7 codepath): pass 142 made the M7 path host-buildable.
- Item 2 (CTest correctness harness over the real init/load/process): tests/arm_core/
  test_arm32_process.c, registered as ctest `arm32_process_correctness`. It also
  wired in spectral_arm32_load + validate_segment_data (passes 145/146 fixed the two
  real embedded bugs the old sim had masked: freq ~7019x too low; amplitude -6 dB).
- Item 3 (sim = perf/resource MODEL over the SAME real code): pass 157. All audio in
  synth_arm32_simulation now comes from spectral_arm32_process; segments converted via
  segment_to_q15 (stretch/pitch/amp folded in, chirp/df dropped) and run through the
  real init/load/process. The retained per-block walk produces only the workload
  accounting (op counts, cache pressure, cycle estimate) — no parallel synthesis.
- Item 4 (one impl only; retire the interim oracle): pass 157 deleted the parallel
  synthesis loop (SimSegment + phase/mix reimplementation). Pass 158 retired
  tests/arm_oracle/ (oracle.py + goldens.json + fixtures): its goldens were stale by
  design once audio moved to the real code, and the compiled CTest harness — which
  oracle.py's own docstring named as its successor — is now the authoritative gate.

Behavioral note: the real loader (validate_segment_data) is now in the sim path, so
the sim is subject to the target's actual constraints (monotonic start+end, <=512
active, chirp rejection). Time-domain output diverges from the desktop-float render
(coarse Q8.8 frequency -> multi-cycle phase drift over a render), but spectral content
matches (dominant frequencies + peak amplitude). This is faithful to the fixed-point
target; CTest, not sim-vs-float, owns correctness.
```

### A2 — Memory-bandwidth redesign (the core of the maintainer's critique)

```text
- Invert the loop nest. Today it is segment-major (one voice across the whole
  block, arm32.c:871), so SPECTRAL_SOA_ACTIVE (on by default) buys nothing.
  Move to sample-major / voice-parallel so SoA lanes are contiguous and the
  accumulator is touched once per sample across voices.
- Wire the real dual-MAC: spectral_smlad() (q15.h:106) is defined but never
  called; the hot loop does 4 independent scalar spectral_mac_q15(). Use packed
  SMLAD / CMSIS DSP on the voice-parallel axis.
- Replace the false "batch LUT lookups minimize cache misses" claim: lookups are
  strided by freq_inc (effectively random in-table). Evaluate phase-bucketed or
  per-voice-register oscillator state instead.
- Quantify with the restricted-profile cycle counters already present
  (SPECTRAL_RESTRICTED_PROFILE); every retained optimization needs a
  benchmark-backed survival (AI_CANON §11, DISCIPLINE_FINDINGS perf).
```

### A3 — Cache coherency contracts (build on pass 136)

```text
- Formalize DTCM/ITCM/AXI-SRAM/SDRAM placement as explicit build contracts for
  segment data, active state, accumulator, and LUT (not comments — §6/§8).
- DMA double-buffering (TODOs.md "Double buffering"): extend the pass-136 DMA
  coherency path to a ping-pong prefetch with explicit invalidate/clean policy
  per buffer cacheability.
- Worst-case execution time and bounded active-segment count as first-class
  outputs, not average benchmarks (DISCIPLINE_FINDINGS, real-time boundedness).
```

### A4 — FPU path evaluation

```text
- Evaluate the fpv5-d16 FPU (embedded_arm_float target already exists) for the
  bandwidth-vs-compute tradeoff raised in TODOs.md (half-word/word mix-in for
  ~1.5x effective bandwidth; interleaving stalled FPU with integer DSP).
- Keep Q15 fixed-point as the default/reference; FPU is an explicitly-gated
  alternative with a parity test, never a silent default.
```

### Phase A closure criteria

```text
- Loop nest is voice-parallel and SoA actually reduces measured cache pressure.
- Every retained optimization has a target-relevant benchmark and a parity test
  vs the scalar reference / interim oracle.
- Placement/coherency/DMA are build contracts with tests, not comments.
- No dead arch code (NEON removed); arch files are backend-specific only.
- A1 defects fixed with regression coverage (interim now, compiled in D5).
```

### Phase A status — A0/A1/A1b DONE; A2/A3/A4 DEFERRED (hardware-gated)

```text
DONE (verifiable on host, no hardware required):
- A0  structural reset / NEON removal           (passes ~138–144)
- A1  correctness defects                        (passes 145/146: freq + amplitude
                                                   bugs fixed, load validation wired)
- A1b verification foundation                    (passes 142/145/146/157/158: real
                                                   M7 host-runnable, CTest harness,
                                                   sim = perf model over real code,
                                                   one impl, interim oracle retired)

DEFERRED — A2/A3/A4 are hardware-gated and cannot be honestly closed in this
environment. Each closure criterion is defined in terms of MEASURED target behavior
(cache pressure, cycle/WCET counts, DMA coherency, FPU-vs-fixed tradeoff), and the
plan itself forbids retaining any optimization without a target-relevant benchmark
(AI_CANON §11). With no Cortex-M hardware and no QEMU cycle model wired up, any A2/A3
rewrite would be unverifiable — it could regress the (now correct, CTest-guarded)
output with no measurement to catch it. The maintainer-set value here is correctness
over unmeasured speed, so we stop at the host-verifiable boundary rather than guess.

  - A2 (memory-bandwidth loop-nest inversion + real dual-MAC SMLAD + LUT-access
       rework): needs measured cache-pressure / cycle counts on the target (or a
       calibrated QEMU model) to prove the inversion actually helps. The sim's
       perf model (pass 157) is the calibration HOOK for this when hardware/QEMU
       numbers exist; until then there is nothing to calibrate against.
  - A3 (DTCM/ITCM/AXI-SRAM/SDRAM placement contracts + DMA double-buffering +
       WCET as a first-class output): needs the real memory map and DMA engine; the
       only consumer is the bare-metal daisy firmware (arm-none-eabi, not a green
       target), mirroring the deferred pass-156 memory-section binding.
  - A4 (fpv5-d16 FPU path evaluation): needs a parity benchmark of the existing
       embedded_arm_float target against the Q15 reference on hardware.

PREREQUISITES IN PLACE for when hardware/QEMU lands: real process is host-runnable
and CTest-guarded (A1b#2); sim emits calibratable MEASURED vs MODELED perf artefacts
over the same real code (A1b#3), so A2/A3 work has a correctness anchor and a
cost-model hook the moment target measurements exist.

Campaign continues into Phase B (contract/guarantee registry), which is host-
verifiable, rather than blocking on hardware.
```

---

## Phase B — Contract / guarantee registry

Objective: a single place that records which kernel guarantees currently hold,
which are relaxed by a flag or runtime path, and how a caller/test can discover
the active set. Generalizes the maintainer's WOLA/COLA concern to all invariants
"hidden behind obfuscations or branches". Home: `core/spectral_contracts.h` +
`docs/core_audit/CORE_CONTRACTS.md`.

### B0 — Establish the COLA/WOLA invariant (prerequisite)

```text
- There is no COLA/WOLA enforcement today: windows are generated unnormalized
  ("normalize explicitly", spectral_windows.h:4-5) and no overlap-add identity
  exists in the engine. A registry cannot record "we broke COLA" until COLA is
  first defined and tested.
- Add the COLA reconstruction invariant (analysis window * synthesis path over
  hop) as a contract with a tested tolerance, citing Griffin-Lim / Allen-Rabiner
  (ACADEMIC_SOURCES) for the WOLA condition.
```

**B0 status (pass 159) — LANDED.** COLA/WOLA invariant defined as header-only
predicates `spectral_overlap_add_envelope_stats` / `_is_constant` in
`core/spectral_contracts.h`, documented in `docs/core_audit/CORE_CONTRACTS.md`
(Griffin-Lim / Allen-Rabiner / Harris cited), and tested by the compiled CTest
`core_contracts` (`tests/core_contracts/test_cola.c`). The test records the key
finding: the engine's *symmetric* (`N-1`) windows do NOT strictly satisfy COLA
(only *periodic* windows at hop N/2,N/4,... and rectangular do; symmetric-Hann
overlap deviation is O(1/N) ~1.5e-3 at N=1024) — acceptable today because the
engine has no overlap-add resynthesis path, so COLA is a latent window property,
not an active runtime guarantee. This is the prerequisite the registry needed: a
defined+tested invariant the B1 manifest can mark as held/relaxed.

### B1 — Guarantee manifest

```text
- Enumerate every correctness/quality-relaxing gate and the invariant it relaxes:
    SPECTRAL_CUSTOM_FAST_MATH_MODE     -> IEEE-754 / determinism
    SPECTRAL_ENABLE_APPROX_TRIG        -> oscillator/window exactness
    SPECTRAL_ENABLE_APPROX_ATAN2       -> phase estimate exactness
    SPECTRAL_ENABLE_APPROX_INV_SQRT    -> magnitude exactness
    SPECTRAL_ENABLE_APPROX_PEAK_LOG    -> peak-interp exactness
    SPECTRAL_METAL_FAST_MATH           -> GPU/CPU parity tolerance
    SPECTRAL_OPT_LEVEL >= 2            -> LUT interpolation (drops to nearest)
    SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS / SPECTRAL_REPRO_BUILD -> reduction determinism
  Each row: invariant id, default state, what relaxing it costs, error budget.
```

### B2 — Self-report

```text
- Compile-time: a single SPECTRAL_ACTIVE_GUARANTEES bitset derived from the gates.
- Runtime: a small query API so a host/test can read the active guarantee set and
  fail closed when an explicit API is handed a descriptor that violates an
  assumed guarantee (DISCIPLINE_FINDINGS, public API).
```

### B1/B2 status (pass 160) — LANDED.

```text
- B1 manifest + B2 self-report shipped as core/spectral_guarantees.h: a 7-bit
  SPECTRAL_ACTIVE_GUARANTEES set (preprocessor-evaluable) + runtime query API
  (spectral_active_guarantees / _guarantee_holds / _guarantees_satisfy fail-closed /
  _guarantee_table). Manifest table in docs/core_audit/CORE_CONTRACTS.md.
- Pinned by CTests core_guarantees (default: exact bits active) and
  core_guarantees_drift (APPROX_* forced on: those bits cleared + each approximation
  asserted within its measured error budget — sin 5e-6, atan2 5e-4 rad, inv_sqrt 1e-5
  rel, peak_log 2e-6). Drift test fails if a polynomial is retuned past budget.
- Two list rows above were corrected, not copied: SPECTRAL_OPT_LEVEL gates nothing
  (unused in C; no bit), and SPECTRAL_REPRO_BUILD is a CMake var driving
  SPECTRAL_CUSTOM_FAST_MATH_MODE (represented through ieee_strict_fp, not its own bit).
- Phase B (B0+B1+B2) is COMPLETE. Campaign proceeds to Phase C.
```

### Phase B closure criteria

```text
- COLA/WOLA invariant exists and is tested.                                  [DONE B0]
- Every relaxing flag is in the manifest with an error budget and a test that
  fails if the flag's documented effect drifts.                             [DONE B1]
- The active-guarantee set is machine-readable at compile and run time.     [DONE B2]
```

---

## Phase C — CTF / KISS adversarial sweep

Objective: adversarially and exhaustively find the "easy" defects — overflow,
sign/type mismatches, narrowing conversions, arithmetic/precedence errors, weak
conditionals, off-by-one, UB — across every file in the kernel until a reviewer
would find nothing embarrassing. Re-runnable with minimal maintainer input.

### Method

```text
- Per-file checklist applied to core/, analysis/, synth/ (and arch files post-A):
    integer overflow / wraparound (esp. size_t<->uint32_t<->uint16_t hops)
    implicit narrowing and float->int casts (use spectral_* checked helpers)
    signedness and shift behavior on negative q15/q31
    fixed-point scaling / saturation correctness
    conditional and loop-bound correctness (off-by-one, >= vs >)
    UB: aliasing, alignment, uninitialized, sequence points
- One pass per file-cluster; each finding -> fix + a pinned regression case.
- Track coverage so the sweep is provably exhaustive and resumable.
```

### Phase C closure criteria

```text
- Every file in kernel/core, analysis, synth swept and signed off.
- Each finding has a fix and a regression case (interim list now; compiled in D5).
- No remaining raw narrowing casts or unchecked allocation arithmetic
  (AI_CANON §17 allocation boundary).
```

### Phase C coverage (resumable sweep log)

```text
SWEPT (one pass per cluster; defects fixed in place + documented in the pass):
  [161] fixed-point UB cluster — synth/backends/arm/spectral_synth_arm32.c,
        synth/math/spectral_q15.{c,h}, core/spectral_lut.h. 4 defects
        (phase-acc signed-overflow/shift UB; amp-ramp mul overflow; phase_rad_to_q15
        out-of-range float->int; portable smlad signed overflow).
  [162] analysis / peak-track cluster — analysis/spectral_peak_track.c and the
        SpectralTracker / fused driver. 3 defects (leak on out-of-range n_threads;
        unclamped thread count; SEVERE inverted return polarity emptying large
        fused-path analysis).
  [163] port/SIMD/out cluster — core/oscillator.c, core/port/{host,embedded}/*.c
        (oscillator_simd, vector_ops, out_kernels), windows/lut/envelope. 3 defects
        (scalar fallback hardcoded FADE_SAMPLES_DESKTOP vs SIMD _ACTIVE — embedded
        scalar/SIMD fade divergence; missing NULL guard on exported magsq_split;
        32-bit ~1U mask on size_t in the M7 stereo-widen kernel).
  [164] hashing/parsing/path cluster — resource_fs (path canon + FNV-1a),
        hash_xx32_xx3 (XXH3/XXH32 adapter), segment_parser, convert_segments,
        segment_{pool,mt}, segment_math. 1 defect (dead divergent duplicate
        validator segment_validate — -Wunused-function + drift hazard vs the
        canonical spectral_segment_payload_valid contract; deleted). Also
        math-reviewed the peak interpolators in spectral_peak_estimator.c
        (Jacobsen/Candan/parabolic/Quinn-second + phase-vocoder) — all correct.
  [165] DSP-math/FFT-scaling cluster + allocation/pool/cache —
        analysis/spectral_analysis_fft.c, core/spectral_windows.c,
        analysis/spectral_peak_{estimator,interp,model}.c, spectral_analysis_fused.c,
        core/spectral_fast_math.c, core/spectral_osc_formulas.h, and
        seg_cache/perf_model/perf{,_embedded}/perf_accounting. 1 defect: the vDSP
        forward real FFT (vDSP_fft_zrip FFT_FORWARD) emits the textbook DFT scaled
        by 2 and that 2x was never compensated, so macOS/vDSP analysis magnitudes
        read 4x (~+6 dB) hotter in magsq than the FFTW/portable branch and the
        contract CTests (which force SPECTRAL_USE_VDSP=0). Fix: 0.5 scale on the
        split (re+im, DC/Nyquist/interior) right after the FFT -> textbook, matching
        FFTW + the PASS8 window amp scales (2/Sigma_window). Phases untouched
        (uniform scale). NOT byte-identical (vDSP code changed) but behaviourally
        inert on every observable output: peak selection is relative to max_magsq
        (scale-invariant — same 657 segs on sin_440hz), frequency uses bin ratios,
        and rendered output is normalized to 0.95 headroom (global scale cancels);
        sim-vs-desktop parity intact (both peak 0.95, RMS within 0.08 dB float/Q15
        gap). The corrected value is the internal/raw magnitude, now == FFTW path.
  [166] synth backends + analysis orchestration — spectral_analysis.c,
        spectral_analysis_full.c, spectral_analysis_fused.c,
        synth/backends/cpu/spectral_synth_cpu.c,
        synth/backends/sim/spectral_synth_simulation.c. 1 defect: in
        spectral_analysis_run_full the window context was freed (struct zeroed,
        descriptor -> NULL) BEFORE spectral_track_peaks_with_window_descriptor read
        window_ctx.descriptor, so the ternary's true branch was statically dead and
        the call always took the HANN fallback. Behaviour-neutral today (type pinned
        to HANN; fallback == same static descriptor) but a read-after-reset and a
        non-HANN mis-calibration landmine (init already takes a window-type param).
        Fix: move the window_context_free below the track-peaks call. Same pointer
        on the HANN path -> identical output (657 segs, peak 0.95, parity intact).
        Rest of cluster verified clean (overflow-guarded shape_init / thread arena;
        sim audio loop's arm32_process return-0 path zeroes output, returns
        num_samples == output_position advance -> no stale data / no stall).
  [167] CLI / orchestration cluster — cmd/cli/spectral_cli.c (arg parse +
        validation), cmd/cli/spectral_cli_pipeline.c (pipeline orchestration), plus
        core/spectral_synth_internal.c, core/spectral_wavetable.c, core/spectral_in.c.
        1 defect: spectral_cli_validate() never enforced the stretch upper bound,
        while SPECTRAL_MAX_STRETCH (=1000.0, spectral_config.h:330) is rejected by
        EVERY stretch consumer (synth_derive_param_scalars; seg_cache:60/136/444;
        segment_parser:52; gpu_tile:42/97) AND by the sibling programmatic validator
        spectral_config_validate:684. So a CLI run with stretch > 1000 passed the
        boundary, ran the full analysis pass, then failed deep in synthesis with a
        misleading PIPELINE_ERR_SYNTHESIS instead of a fast boundary rejection. Fix:
        reject stretch > SPECTRAL_MAX_STRETCH in spectral_cli_validate (inclusive cap,
        mirroring spectral_config_validate's two-step common+cap structure).
        Behaviourally inert for every in-contract stretch (0 < stretch <= 1000):
        desktop+sim still 657 segs / peak 0.95 on sin_440hz, boundary stretch=1000
        accepted (exit 0); over-cap stretch=2000 now fails fast with a clear message,
        exit 1, no analysis. NOTE: the cache-mode unguarded out_len casts
        (pipeline lines 938/975) are NOT a reachable UB — build_cache_key returns 0
        for stretch>1000 (seg_cache:444), disabling cache mode before those lines, so
        with stretch bounded the product cannot approach SIZE_MAX (left unguarded by
        design). synth_internal/wavetable/audio-I/O all audited clean this increment.
  [168] Embedded Q15 fade envelope — synth/backends/arm/spectral_synth_arm32.c.
        1 defect: the linear fade ramp used the FIXED constant SPECTRAL_FADE_STEP_Q15
        (= Q15_MAX/SPECTRAL_FADE_SAMPLES_EMBEDDED = Q15_MAX/32 = 1023) as the per-sample
        slope, but the activator clamps fade_len to seg_length/2 (down to 1) for
        segments < 64 samples. So a short segment's fade-IN stopped at
        (fade_len-1)*(Q15_MAX/32) instead of reaching full scale, while the fade-OUT
        always seeds at Q15_MAX -> amplitude discontinuity (click) at the fade
        boundary and a large non-zero value at segment end. The desktop reference
        (spectral_envelope.c) correctly uses 1/fade_len. Fix: derive a per-segment
        fade_step = (q15_t)(Q15_MAX / fade_len) in BOTH the M7 (synth_segment_m7) and
        generic/scalar paths and substitute it at all 8 use sites; removed the now-dead
        SPECTRAL_FADE_STEP_Q15 macro (spectral_q15.h) to kill the divergent-duplicate
        hazard. For fade_len==32 (every segment >= 64 samples) Q15_MAX/32 == 1023
        EXACTLY, so long-segment output is byte-identical (arm32_process_correctness
        golden passes unchanged). Added tests/arm_core test_short_segment_fade: pins
        the oscillator at peak (freq=0, phase=quarter turn, amp=1.0) so each sample
        reads back the envelope, asserts a len=32/fade_len=16 segment reaches full
        scale at the midpoint with a small boundary jump (fade_in_end=0.935 peak=0.998
        fade_out_end=0.063 jump=0.063; pre-fix would be ~0.47/0.53). Flagged for pass
        169 (now fixed): spectral_synth_internal.c:84 synth_zero_output_if_valid unused.
  [169] Core synth dispatch / shared helpers — core/spectral_synth_internal.c +
        core/spectral_backend.c. 1 defect: synth_zero_output_if_valid (line 84) was a
        defined-but-never-called static (zero call sites tree-wide) that duplicated the
        output-zeroing already done inline by synth_preflight_common via the once-
        computed, overflow-checked preflight_out_bytes (strictly cheaper). It warned
        -Wunused-function on simulate_daisy/embedded_arm/embedded_arm_float (host dead-
        strips it silently) and is a drift hazard (cf. pass 164's dead validator). Fix:
        delete it. Behaviour-neutral (uncalled static -> no reachable code). Rest of
        cluster audited clean: param derivation (stretch underflow + MAX cap), preflight
        zeroing across all early returns, segment_loop_params_init overflow/finite
        gauntlet, the one-shot GPU tile/seg caches (cleared on every read/miss so no
        stale-pointer reuse), gpu_dispatch_plan_init (array-bytes guards + free-on-
        error), and the spectral_backend.c vtable dispatch (total fallback, n_threads
        floor, GPU+wavetable/unavailable/unsupported-timbre/synth-fail all -> CPU).
        Verified: 5 builds clean (warning gone) + ctest 4/4 + 340-seg parity.
  [170] Binary deserialization / converter cluster — core/spectral_seg_cache_fs.c,
        core/spectral_seg_cache.c, core/spectral_segment_parser.c, cmd/convert_segments.c
        + the contract validators (spectral_contracts.h) and primitives
        (spectral_endian.h, spectral_size_*/array_bytes). 1 defect: convert_segments.c
        loaded the SPEC .bin via raw fread and NEVER byte-swapped the header or
        segments, violating the documented "files are always little-endian; swap on
        load" contract that every sibling loader honors (segments_load via
        header_from_le + spectral_segment_swap_endian; the seg cache via
        seg_cache_*_swap). On a big-endian host the version field would byte-reverse
        and the tool would reject every valid file (fails-safe, but unusable). Fix:
        added a single-source-of-truth `spectral_segment_file_header_swap_le` inline to
        spectral_segment_parser.h (header-only, since convert_segments does NOT link
        spectral_segment_parser.c), deduped the parser's static header_to_le/from_le to
        call it, and swapped header+segments on load in convert_segments. Every swap is
        guarded by spectral_is_big_endian(), so the change is a provable no-op on LE:
        the converted .spq is BYTE-IDENTICAL pre/post-edit (cmp on a 340-seg fixture)
        and the desktop/simulate binaries are behaviour-unchanged. The rest of the
        cluster audited CLEAN: seg_cache_fs index load (exact file-size vs count check
        before malloc), seg_cache_entry_metadata_valid + seg_cache_validate_data_extent
        (overflow-guarded data_offset+bytes <= file size), seg_cache_validate_tile_blob
        + spectral_gpu_tile_layout_words_valid (contiguous-range + exact-ref-count +
        per-id<seg_count bound), spectral_segment_array_payload_valid /
        _gpu_array_matches_segments (finite gauntlet + GPU/base field equality),
        spectral_omega_to_q88 (>255 /4-encode then clamp 255 -> max 65280 < 65536, no
        overflow) / spectral_phase_rad_to_q15, and the endian + size_mul/add/array_bytes
        primitives (builtin-overflow with manual fallback). Verified: 5 builds clean +
        ctest 4/4 + convert_segments .spq byte-identical + 340-seg desktop parity.

  [171] GPU tile preprocess concurrency — core/port/host/spectral_gpu_tile.c. 1 defect:
        the histogram `#pragma omp parallel` (no num_threads clause) indexes
        thread_counts[omp_get_thread_num()], but thread_counts is sized to
        spectral_omp_effective_thread_count() == min(omp_get_max_threads(),
        SPECTRAL_MAX_THREADS=256). On a host where omp_get_max_threads() > 256 (>256-thread
        machine or OMP_NUM_THREADS>256, both user-reachable), the unqualified region spawns
        omp_get_max_threads() threads, so tid in [256,max) reads thread_counts[tid] out of
        bounds (garbage pointer) then writes through it (my_counts[tt]++) — heap OOB R/W.
        Inconsistent with every sibling region in the codebase, all of which pin the team
        (peak_track.c num_threads(n_threads), analysis_fft/fused, synth_cpu). Fix: added
        num_threads(n_threads) to the histogram region so tid is bounded by the array size.
        The second region (fill, atomic-capture into tile_cursors, indexes no per-thread
        array) is correct at any team size and left untouched (KISS). On the LE/<256-thread
        verification host n_threads == omp_get_max_threads(), so the clause requests the
        SAME team the default already used → provably inert. Verified: 5 builds clean +
        ctest 4/4 + cache-mode GPU-tile output BYTE-IDENTICAL HEAD-vs-fixed (340-seg
        sin_440hz fixture; the path is deterministic run-to-run, confirmed first).

  [172] Canonical oscillator math contract — core/spectral_osc_formulas.h (+ mirrored
        Metal MSL in core/oscillator.c). 1 defect: spectral_osc_asin fed
        spectral_normalize_phase(p)*INV_PI straight into asinf() with no domain clamp.
        normalize_phase aims for [-pi,pi) but in IEEE-754 single precision the inexact
        TWO_PI/INV_TWO_PI constants + the p - TWO_PI*k rounding let the result land a
        sub-ULP below -pi (overshoot grows with |p| via catastrophic cancellation), so
        rads*INV_PI drops just below -1.0 and asinf returns NaN — which flows out as the
        segment wave and poisons the synth (dst[j] += amp*wave), then the downstream
        peak-over-buffer normalization can propagate one NaN to the whole rendered file.
        ASIN (TIMBRE_ASIN=4, user-selectable) has no SIMD variant so it ALWAYS takes the
        scalar osc_asin path. Fix: clamp rads*INV_PI into [-1,1] before asinf (endpoints
        map to the correct +/- pi/2 boundary value) in the shared header (covers CPU
        scalar AND CUDA, which includes the header directly), mirrored in the Metal MSL
        string (asin(clamp(...))); bumped SPECTRAL_OSC_FORMULAS_VERSION 4 -> 5 across all
        three _Static_assert guards (the metal .m guard correctly tripped the build until
        updated). The other 7 generators audited clean — only asin carries a bounded
        domain; normalize_phase deliberately NOT touched (a hard clamp would alter rads
        for every backend/waveform; fix the one consumer that cannot absorb the overshoot,
        not the shared producer). Inert for in-domain inputs (|arg|<=1 path identical;
        non-ASIN timbres never call it). Reachability proven at function level (pre-fix
        262763 NaN; FIXED 0 NaN over 409.6M swept phases; nextafterf(-pi) -> -pi/2) and on
        a realistic segment phase0 + j*(alpha+beta*j) trajectory with real musical freqs
        (pre-fix 330 NaN vs 0 over 48.4M calls, ~7/million). Verified: 5 builds clean +
        ctest 4/4 + SINE out_c.wav BYTE-IDENTICAL HEAD-vs-fixed (non-ASIN inert) + base
        ASIN byte-identical (fixture misses the rare band) + aggressive pitch/stretch
        float-WAV NaN scan = 0.

  [173] Host SIMD oscillator port — core/port/host/oscillator_simd.c. 1 defect:
        wave_quantized_4 (the SSE/SIMDe quantized sustain lane) dropped the canonical
        spectral_osc_quantized() domain guard. The scalar contract returns 0 when
        scaled = rads*width is non-finite or outside [INT_MIN,INT_MAX]; the SIMD lane
        relied on simde_mm_cvttps_epi32, which is DEFINED to return INT_MIN (0x80000000)
        for out-of-range/NaN — not 0 — so it emitted INT_MIN*inv_w (an out-of-[-1,1]
        value, ~-2.147 at width~1e9) where the scalar yields 0. rads is normalized to
        [-pi,pi], so scaled overflows int only when |width| > INT_MAX/pi ~ 6.84e8. The
        analysis path hard-codes width=0.5 (spectral_peak_interp.c) so it never hits the
        band; the large-width band is reachable via deserialized .seg segments, whose
        validation (spectral_segment_array_payload_valid -> isfinite(width), contracts.h:42)
        checks finiteness but NOT magnitude. Divergence is also intra-segment: fade-in/out
        + sustain-tail use the scalar lane wave_quantized_1 -> spectral_osc_quantized
        (guarded -> 0) while the sustain body used the unguarded wave_quantized_4. Fix:
        AND an in_range mask (scaled>=(float)INT_MIN & scaled<=(float)INT_MAX) onto the
        result; the >=/<= comparisons also reject NaN/Inf (all NaN compares false),
        reproducing the canonical guard exactly. SIMD pwm lane (wave_pwm_4) audited and
        left unchanged — it only does a comparison -> +/-1 (no int cast), and its missing
        !isfinite guards are already enforced upstream by payload_valid, so no reachable
        divergence (KISS: no speculative no-op). Embedded SIMD port routes quantized to a
        scalar stub (osc_simd_available excludes it) so the canonical guard already applies.
        Inert for in-range lanes (mask all-ones -> bit-identical) and for non-quantized
        timbres (never call it). Reachability proven at function level: pre-fix vs post-fix
        vs canonical scalar over a 4,000,001-pt rads sweep at several widths — width=0.5/1.0
        0 divergences both; width=1e9 pre-fix 1,265,743 (worst |d|=2.147) -> post-fix 0;
        width=1e30/FLT_MAX pre-fix all-diverge -> post-fix 0. Verified: 5 builds clean +
        ctest 4/4 + quantized (timbre=6) out_c.wav BYTE-IDENTICAL HEAD-vs-fixed on
        sin_440hz backend=cpu (realistic width=0.5 pipeline is inert).

  [174] File-I/O + CLI untrusted-input boundary cluster — CLEAN AUDIT (no defect, no code
        change). Audited core/spectral_in.c (libsndfile reader + channel downmix + time
        window), core/spectral_windows.c (Hann/Hamming/Blackman/rect generators + window
        metrics + Smith log-parabolic peak interp/height), runtime/spectral_utils.c numeric
        parsers (parse_i32/f32/size_arg, getenv_f64*), cmd/cli/spectral_cli.c validation
        (n_fft power-of-two>=64, hop in [1,n_fft], stretch finite-positive<=MAX, pitch
        finite+range, timbre [SINE,PWM], backend enum, threads clamped), core/spectral_out.c
        (float-WAV writer + bounds-checked RIFF PEAK-timestamp scrubber). Findings: every
        untrusted quantity is overflow/finiteness-guarded before use; reader rejects non-
        finite samples pre-math; writer rejects non-finite pre-write; window generators guard
        N==0/N==1 (no /(N-1) div-by-zero); Smith interp verified algebraically exact
        (p=0.5*(y[-1]-y[1])/(y[-1]-2y[0]+y[1]), y=log power) with [-0.5,0.5] clamp on both
        branches; metric scales derived from the realized window (backend-convention safe);
        parsers fully check errno/no-digits/trailing/range/finite (no atoi/atof UB); RIFF
        scrubber bounds-checks payload/chunk extents, odd-size padding (UINT64_MAX guard),
        and the size==0 chunk still advances 8 bytes (no infinite loop). No SPECTRAL_MAX_FFT
        cap exists but downstream sizing is spectral_size_mul-guarded so it is not a gap.
        Verified: no source changed -> Pass 173 green state preserved (5 builds clean, host
        binaries byte-identical); ctest re-run 4/4.
  [175] Peak frequency-estimation cluster — CLEAN AUDIT (no defect, no code change).
        Audited analysis/spectral_peak_estimator.c (Jacobsen/Candan/Quinn-second complex
        estimators, log- & magnitude-parabolic interp, phase-vocoder advance, magsq->amp +
        bounded window-peak gain), analysis/spectral_peak_interp.c (validate_candidate
        neighborhood load), the candidate generators + freq_step_df derivation in
        analysis/spectral_peak_track.c, and the omega/df/da consumer in
        core/spectral_segment_math.h. Findings: (1) the open OOB lead is closed — both
        candidate generators start f=1 with SIMD bound f+7<n_freqs-1 / scalar f<n_freqs-1,
        so every cf in [1,n_freqs-2] and validate_candidate's row[cf-1]/row[cf+1] loads are
        always in-bounds (the missing cf+1<n_freqs check is a generator invariant; the
        cf==0 guard protects the public surface). (2) every estimator matches its published
        form term-for-term: Jacobsen Re{(X[k-1]-X[k+1])/(2X[k]-X[k-1]-X[k+1])}; Candan
        *tan(pi/n_fft)/(pi/n_fft); Smith mag-parabolic 0.5(a-c)/(a-2b+c); Quinn-second
        d=0.5(dp+dm)+tau(dp^2)-tau(dm^2) with dp=-ap/(1-ap),dm=am/(1-am) and exact tau
        constants; phase-vocoder residual=princarg(dphi-k*step_omega*hop). (3) the suspicious
        0.5 in freq_step_df is CORRECT — phase model phi(n)=phi0+omega*n+df*n^2 gives
        d omega/dn=2*df, so df=Delta omega/(2*hop)=bin_delta*0.5*step_omega/hop, passing the
        correct endpoint frequency at n=hop (da is the linear analogue, no 0.5). All products
        formed in double with finite + |.|<=FLT_MAX + denom-epsilon guards; offsets clamped
        [-0.5,0.5]; edge best_next degrades to no-refinement, never OOB. Verified: no source
        changed -> Pass 174 green state preserved (5 builds clean, host binaries
        byte-identical); ctest 4/4.
  [176] SpectralTracker lifecycle / per-thread storage / OpenMP reduction — CLEAN AUDIT
        (no defect, no code change). Audited spectral_tracker_create/destroy/
        free_segment_storage, emit_segment per-thread grow+store (spectral_peak_interp.c),
        spectral_tracker_process OMP pair loop, spectral_tracker_finalize prefix-sum merge,
        frame_time_from_index, the spectral_aligned_alloc/free pairing, and the
        TrackSegment(32B)/Segment(64B) layout. Findings: all sizing overflow-checked
        (spectral_size_mul/add); seg_arrays calloc'd so the goto-fail path free()s safely;
        consistent seg_arrays[tid] (plain) vs seg_counts/capacities[tid*STRIDE] (padded)
        indexing everywhere; spectral_aligned_alloc wraps C11 aligned_alloc (free-compatible)
        with overflow + size==0 guards; per-thread grow is single-writer race-free with
        doubling + wrap/mul overflow guards; process() next_row is NEVER NULL (overlap else-
        branch only reached when overlap_magsq_row!=NULL) and next_phase_row=NULL is safe
        (only used by phase-advance which guards !next_phase_row; default policy IGNORE);
        global_frame_offset+t overflow pre-checked; finalize prefix-sum offsets make the
        parallel copy ranges disjoint (race-free), total_segs>UINT32_MAX rejected, segs/offsets
        NULL-init so fail-path free is safe and all goto-fail precede the success free (no
        double-free); the memcpy(sizeof(TrackSegment)=32)+memset(_pad_w,32) lands every field
        in its matching Segment slot (width@28 set by memcpy, not clobbered) and fully
        initialises the 64-byte record (no heap-leak), enforced by _Static_assert. Verified:
        no source changed -> Pass 175 green preserved; full triad re-run (5 builds clean,
        ctest 4/4).
  [177] STFT analysis FFT driver + orchestration cluster — CLEAN AUDIT (no defect, no
        code change). Audited spectral_analysis_fft.c (vDSP + FFTW frame transform, one-
        sided magsq scaling, vDSP 0.5 rescale, phase, OMP reduction(max), alloc/free),
        spectral_analysis.c (frame-count/shape, path decision, scale wiring),
        spectral_analysis_full.c, spectral_analysis_fused.c (two-pass concurrent path),
        spectral_processing_chain.c (mask parse/dispatch), spectral_peak_model.c (policy
        resolution) + the proc_* no-op stubs; traced consumer/producer contracts into
        spectral_windows.c (magsq-scale = amp_scale^2), spectral_vector_ops.c
        (magsq_only/_phase) and the spectral_peak_track.c threshsq dB math. Findings: the
        one-sided convention is applied EXACTLY ONCE (both backends compute raw re^2+im^2;
        only apply_magsq_scales doubles interior bins and recomputes interior-only frame
        max); positive magsq scale = (2/Σ)^2 vs endpoint (1/Σ)^2 (4:1 ratio is the
        amplitude-recovery convention so sqrt(scaled)=A); the vDSP ×0.5 exactly undoes
        vDSP_fft_zrip's 2x (uniform on re+im => phase-preserving) and DC=realp[0]^2 /
        Nyquist=imagp[0]^2 / interior vsq+vadd over mid=n_freqs-2 (n_fft>=64 => no
        underflow, loads in bounds, all n_freqs bins written once); phases match the FFTW
        atan2(im,re) branch incl. DC/Nyquist sign->0/π; n_frames=(n_samples-n_fft)/hop+1
        keeps the last frame in bounds; fused frame-pair contract holds by induction
        (row=magsq[pair], next=magsq[pair+1], t_hop=pair*hop) with tid-indexed per-thread
        storage bounded by num_threads(actual_threads)<=tracker slots, atomic time accum,
        reduction(max) global max over all frames matching the full path; threshsq uses
        pow(10,dB/10) — the POWER convention (/10 not /20) correct for magsq; parse/
        dispatch/policy paths have no memory or arithmetic hazards (snprintf truncation
        bounded, strtoul overflow caught by ALL_KNOWN mask). Verified: no source changed
        -> Pass 176 green preserved; full triad re-run (5 builds clean, ctest 4/4).

  [178] CPU additive-synthesis + wavetable + oscillator-math cluster — DEFECT FIXED
        (host SIMD PWM non-finite divergence). Audited spectral_synth_cpu.c (thread-buffer
        arena, parallel-for partitions, float/native reduce, segment callbacks),
        spectral_synth_internal.c preflight (double** t_synth null-redirect to dummy),
        spectral_envelope.c (raised-cosine fade), spectral_wavetable.c (builtins, .spwt
        load/save w/ format conversion, load_raw, Intel-HEX load, load_buffer, lookup_f/_q),
        spectral_osc_formulas.h (8 canonical waveforms + phase normalize + fade), oscillator.c
        (timbre_table dispatch + scalar/SIMD segment), and both oscillator_simd.c ports.
        DEFECT: core/port/host/oscillator_simd.c wave_pwm_4 did not mirror the canonical
        spectral_osc_pwm() domain guard (!isfinite(rads)||!isfinite(width) -> 0). Its sibling
        wave_quantized_4 was fixed for exactly this class (Pass 173) but PWM was missed. On the
        desktop float build the live PWM path is segment_fn_timbre -> timbre_synth_segment ->
        osc_simd_segment_pwm -> osc_simd_fused_sustain(wave_pwm_4 [sustain], wave_pwm_1 [fade]).
        For a non-finite phase (reachable: spectral_segment_payload_valid bounds width finite
        but leaves omega unbounded above, so a deserialized segment's finite-but-huge omega
        overflows the accumulated phase to +/-Inf -> NaN after normalize_phase), the sustain
        lane emitted +/-1 while the fade lane and the canonical emit 0 — a seam inside one PWM
        segment + a contract divergence. FIX: rewrote wave_pwm_4 to (a) return 0 for non-finite
        width, (b) keep width<=0 -> 1, else +/-1 via the threshold compare, then (c) AND every
        lane with a finite-rads mask (|rads|<=FLT_MAX rejects NaN [unordered] and +/-Inf),
        forcing non-finite lanes to 0 in all four width corners — matching the scalar contract
        exactly. Embedded port unaffected (quantized/pwm are scalar-fallback stubs there).
        CLEAN elsewhere: wavetable file loaders are fully bounds-checked (HEX offset/data_len
        guarded, covered_bytes==expected_bytes, samples[SIZE]=samples[0] wrap guard, payload
        sizing per file-format); native-reduce missing finite-check is an unreachable non-defect
        (synth_cpu_native has zero callers; embedded Q15 is finite-by-saturation). Verified by
        VALUE PARITY not byte-identity (host code changed): for every finite input the finite
        mask is all-ones so output bits are unchanged; only the degenerate non-finite PWM lane
        converges to 0. Full triad: 5 builds clean, ctest 4/4.

  [179] GPU synthesis dispatch cluster + cross-backend timbre gate — CLEAN AUDIT (no defect,
        no code change). Audited gpu_timbre_supported / gpu_check_timbre_or_fallback,
        spectral_gpu_dispatch_plan_init/_free, gpu_tile_preprocess_cached, gpu_seg_cache_*,
        gpu_synth_params_pack_checked, and the Metal host dispatch (synth_metal buffer
        build/upload) + the Metal MSL oscillator() string. The lead (MSL switch handles only
        timbres 0-5, omitting QUANTIZED/PWM) is NOT a defect: gpu_timbre_supported gates exactly
        0..5 (TIMBRE_PARABOLA) so 6/7 fall back to the CPU synth before any GPU dispatch — the
        6-waveform kernel is never asked to render a width-based timbre. Plan construction is
        overflow-checked (segment_bytes/tile_ids_bytes/tile_ranges_bytes via spectral_array_bytes;
        sa.count is uint32 so the seg-cache pass cannot truncate); the seg cache is single-use w/
        exact-count match; _free is owns-flag-guarded (no double-free). Boundary pack rejects
        out_len/num_segments > UINT32_MAX, timbre outside [SINE,PWM], and non-finite-positive
        stretch factors. Metal dispatch handles NULL segment_source (packs from sa) and sizes
        every buffer from the checked plan fields. Verified: no source changed -> Pass 178 green
        preserved; full triad re-run (5 builds clean, ctest 4/4).

  [180] Segment-cache persistence cluster (core/spectral_seg_cache.c) — CLEAN AUDIT (no defect,
        no code change). Audited the full lookup + store paths plus helpers: seg_cache_key,
        seg_cache_bsearch (Java ~insertion convention), seg_cache_tile_blob_bytes (overflow-checked
        sizing), seg_cache_entry_metadata_valid (per-field disk validation), validate_data_extent
        (file-size bound), validate_tile_blob (header-vs-index cross-check + layout validation).
        Every disk-sourced field is validated before it is narrowed (sample_rate/stretch/pitch
        range, output_length uint64<->size_t round-trip, seg_count/tile_count cross-invariants);
        total_data_bytes is folded via spectral_array_bytes + spectral_size_add then bounded by the
        real data-file size (data_file_size >= data_offset + total, with a UINT64_MAX-data_offset
        pre-add guard) so the mmap/read is provably in-file. Both the mmap fast path and heap
        fallback re-validate the Segment payload AND the packed GPU-mirror before accepting, and
        reject-with-unmap on corruption; result_free is mmap-aware. Tile blobs are only published
        after the on-disk header cross-checks tile_size/num_tiles/total_refs against the index entry
        and spectral_gpu_tile_layout_words_valid passes. Store validates-before-write, appends
        transactionally (begin/write/end with abort-on-error so a half-written record never updates
        the index), big-endian-swaps into scratch with per-record scalar fallback, and the sorted
        index insert is overflow-guarded (ins>count -> FILE_CORRUPT; new_count>UINT32_MAX rejected;
        the (uint32_t)(count+1) wrap is documented + avoided). Verified: no source changed -> Pass
        179 green preserved; full triad re-run (5 builds clean, ctest 4/4).

  [181] Q15 ARM32 fixed-point synthesis cluster (synth/backends/arm/spectral_synth_arm32.c +
        synth/math/spectral_q15.h) — DEFECT FIXED. The active-segment list pruned expired segments
        in the PROCESSING loop, which runs AFTER the activation scan. The loader
        (spectral_arm32_validate_segment_data) bounds simultaneous-active vs SPECTRAL_ARM32_MAX_ACTIVE
        (512) with a HALF-OPEN overlap model (first_end > start: a segment ending exactly at a new
        segment's start does NOT count), so an input with 512 segments ending at sample X plus a new
        segment starting at X is valid. At runtime, at the block with out_pos==X those 512 expired
        entries still occupied slots during activation -> num_active==512 -> the new segment failed the
        `num_active < MAX_ACTIVE` gate and was dropped (lost partial/onset) for a config the loader
        accepted. Reachable for dense, hop-aligned spectra. FIXED: added
        spectral_arm32_prune_expired_active(ctx, out_pos) (swap-with-last, SoA+AoS) called BEFORE the
        activation loop so occupancy matches the validated half-open model; deleted the now-redundant
        processing-loop removal (post-prune every active provably has seg_end > out_pos). spectral_q15.h
        is clean (float<->Q15/Q31 clamp at +/-1; phase_rad_to_q15 keeps n in [0,1); portable smlad wraps
        in uint32 to match ARM non-saturating MAC; mul_q15 saturates -1*-1). Verified: embedded-only
        change (file is inside #if SPECTRAL_EMBEDDED) so desktop binary byte-identical; functional parity
        below the 512 ceiling (prune removes exactly the same seg_end<=out_pos entries) so
        arm32_process_correctness stays green; only the at-ceiling boundary-aligned case changes (a
        previously-dropped partial now renders). Full triad: 5 builds clean, ctest 4/4.

  [182] Host file-I/O layer (core/spectral_fs.c + core/spectral_seg_cache_fs.c) — CLEAN AUDIT (no
        defect, no code change). The primitives that back the seg cache: u64<->off_t conversion,
        open/close/seek/tell/file_size, read_exact[_path], write_exact, map_ro_path (mmap + page
        alignment), and the index load/write + data append wrappers. Every offset+bytes is
        overflow-guarded (bytes > UINT64_MAX - offset) then bounded by the real file size before any
        seek/read/mmap; the mmap page-offset math is overflow-checked (bytes > SIZE_MAX - page_off),
        map_start round-trips through u64_to_off, and map_len = page_off + bytes is provably within
        the file from the page-aligned start so no access faults past EOF (no SIGBUS); the fd is
        closed on every error path (no leak). The index loader validates file_size == header +
        count*entry EXACTLY before the malloc (a corrupt huge count cannot blow up the allocation);
        append is transactional (begin/write/end/abort) and 64-bit-offset clean via fseeko/ftello.
        Verified: no source changed -> Pass 181 green preserved; full triad (5 builds clean, ctest 4/4).

  [183] CUDA tile-parallel synth backend (synth/backends/gpu/cuda/spectral_synth_cuda.cu) — DEFECT
        FIXED. In synth_cuda the GPU-timing local `float gpu_ms = 0.0f;` was declared AFTER ~15
        `goto cleanup` statements whose target label lies inside that local's scope, so every goto
        jumped into the scope of an *initialized scalar* — ill-formed C++ ([stmt.dcl]/3 exempts the
        jump only for a scalar declared WITHOUT an initializer), which nvcc (host code = C++) rejects
        with "jump to label 'cleanup' crosses initialization of 'float gpu_ms'". Latent because the
        macOS build host has no CUDA toolkit (the .cu is in no production target) and the Metal
        sibling carries the same late local but compiles as Objective-C, where the jump is legal.
        FIXED by hoisting the declaration above the first goto (top-of-function block) and dropping
        the redeclaration — pure well-formedness fix, runtime behavior byte-identical. The kernel DSP
        math bit-matches the Metal/CPU canonical formulas (seg bounds test, alpha/beta/d_amp/phase/amp
        via spectral_segment_*_f32, fade_envelope via spectral_fade_envelope_gpu), __syncthreads is
        reached uniformly (range.count is per-tile, equal across the block), the cooperative load is
        bounded (TILE 512 threads >= SEG_CACHE 256), and the host buffer/stream/event lifecycle frees
        on every error path. Verified: .cu in no host target so binaries unaffected; full triad
        (5 builds clean, ctest 4/4).

  [184] Segment storage + Q15 sine LUT + CPU fade-envelope + resource bridge (core/spectral_segment_
        pool.c, spectral_segment_mt.c, spectral_envelope.c, spectral_lut.c/.h, spectral_resource_
        bridge.c) — CLEAN AUDIT, no defect, no code change. Block-chain segment pool never reallocs
        payload (Segment* stay stable), grows the block-pointer array by doubling under a max_blocks
        guard, and to_array copies exactly `count` with per-memcpy overflow checks. The MT array holds
        sa->mutex on every mutator, get() returns a documented SHALLOW borrow while copy() deep-copies
        under an overflow guard, and destroy() frees both arrays then the mutex (no double-free/leak).
        The CPU fade indexes by integer j in [0,len-1], so fade_envelope_out's from_end = len-1-j is
        always >= 0 and the out-ramp is monotone to 0 at the final sample. The Q15 LUT fills SIZE+1
        entries (index SIZE = wrap guard), lut[idx+1] tops out at the guard (no OOB), the 8-bit frac
        weight makes weight/256 == frac_raw/16 exactly, the q31 interp product can't overflow int32
        and stays within [s0,s1], and cos = sin(phase+16384) is the exact quarter-turn shift. The
        resource bridge is a benign zero-count stub. Independently re-derived the peak-estimator/
        window/oscillator DSP math (162-178) clean. ONE bounded, cross-backend-consistent GPU
        observation recorded but deliberately NOT changed: spectral_fade_envelope_gpu takes a float
        sample offset, so under time-stretch (fractional seg_start) j can land in (seg_len-1, seg_len),
        making from_end in (-1,0) and the GPU fade-out non-monotone (swings back toward ~1 at the
        worst case) — bounded in [0,1], no overflow/NaN/OOB, IDENTICAL on CUDA + Metal (parity contract
        holds), and a fix would require a coordinated header + Metal-MSL-string + SPECTRAL_OSC_FORMULAS_
        VERSION bump that CANNOT be verified on this host (ctest exercises ARM/contract paths, not GPU
        fade output). Deferred to a maintainer-directed, test-backed change (natural fit for Phase D's
        golden-vector loop). Verified: no source changed -> Pass 183 green preserved; triad last green
        at end of Pass 183 (5 builds clean, ctest 4/4).

  [185] Support/utility math cluster (core/spectral_hash_xx32_xx3.c, spectral_resource_fs.c,
        spectral_common.c, runtime/spectral_utils.h, runtime/spectral_perf_model.c, synth/math/
        spectral_q15.c+.h, core/spectral_error.c, analysis/spectral_peak_model.c) — CLEAN AUDIT,
        no defect, no code change. These were the files the 161-184 sweep touched only incidentally
        (0-2 patch-note mentions each). The resource-path canonicalizer was re-derived BYTE-FOR-BYTE
        against the pure-Python reference (compress_path in resource_hash_reference.py): all five
        phases match including (a) the phase-1 truncation boundary — Python's MAX_BYTES = PATH_SIZE-1
        = 1023 is exactly C's reserved null terminator, so both cap at 1023; (b) the NTFS trailing-
        space-before-".." and trailing-dot/all-dots component rules; (c) the run-of-256 RLE remainder
        producing token(255)+literal(1) on both sides with no RLE->literal fallback for runs>=2; and
        (d) FNV-1a basis/prime. The Q15 primitives are saturation/rounding/overflow correct: smlad's
        portable fallback uses uint32 wraparound (two Q15 products can sum to exactly 2^31 = INT32_MAX
        +1, so a signed accumulate would be UB) matching ARM __smlad; mul_q15 saturates Q15_MIN^2 ->
        32767; phase_rad_to_q15 defends the n+=1.0f-rounds-to-1.0 case; q30_to_q15 uses >>15 (not the
        pass-145 -6 dB >>16 bug). size add/mul guard every multiply (`a!=0` before SIZE_MAX/a),
        aligned_alloc guards the round-up overflow, the hash full-direct path guards subtraction +
        int64 cast + size_t narrowing, the perf model is uint64/threshold-guarded, and error/peak-model
        are table/policy code with no arithmetic. Verified: no source changed -> Pass 184 green
        preserved; re-ran full triad (5 builds clean, ctest 4/4).

  [186] Synthesis backend dispatch + wavetable bank (core/spectral_backend.c,
        core/spectral_wavetable.c) — CLEAN AUDIT, no defect, no code change. Wavetable lookups
        compute idx/frac/guard correctly: lookup_f computes frac BEFORE the `idx>=SIZE -> 0` clamp
        so the float-rounding phase->1.0f case (idx_f==SIZE) returns samples[0] with frac 0, and
        otherwise samples[idx+1] tops out at the SIZE wrap-guard slot; lookup_q's `frac <<= 3`
        gives max 248/256 = 31/32 (correct exclusive-upper weight for the 5-bit fraction at
        BITS=11), verified across all frac_bits regimes. The .spwt/raw/HEX/buffer loaders are
        overflow-checked (spectral_array_bytes/size_add), exact-file-size-validated, finite-
        validated, and leak-free on every error path; the Intel-HEX parser enforces byte_count <=
        data_capacity, exact line length (so data+checksum reads stay in-line), and the two's-
        complement checksum, while load_hex bounds every write by offset+data_len <= expected_bytes
        (no underflow — offset>expected rejected first) with per-byte coverage tracking and a
        required EOF record. The backend dispatch is a correct single-direction CPU-fallback cascade
        (not-compiled / unavailable / unsupported-timbre / GPU+wavetable / synth-error all route to
        CPU) with a bounds-guarded name table (`idx <= BACKEND_EXPORT`). Verified: no source changed
        -> Pass 185 green preserved; triad re-run green this session (5 builds clean, ctest 4/4).

  [187] ARM DWT/ITM debug instrumentation + analysis-proc stubs + processing-chain/console/log/
        audio-in (synth/backends/arm/spectral_debug_embedded_arm.c, analysis/spectral_proc_{serra_
        smith_1990,johnston_1988,adaptive_track_density}.c, analysis/spectral_processing_chain.c,
        runtime/spectral_console.c, core/spectral_log.c, core/spectral_in.c) — DEFECT FIXED (x3) +
        rest CLEAN. Fix: the debug monitor's three exponentially-weighted rolling averages used the
        embedded idiom `avg += (new - avg) >> k` on uint32_t fields (cycles_avg, read/write_latency_us).
        That idiom is only correct in signed arithmetic; in unsigned, a below-average sample (the
        common case — any block faster than the running mean) makes `(new - avg)` underflow to >=2^31
        and the logical `>>` adds ~2^28 instead of subtracting, corrupting the mean. Fixed all three
        sites (timing_end >>4, sdcard_read/_write >>3) to form the delta in int32_t and fold back into
        the uint32_t field; the averaged cycle/us quantities are << INT32_MAX so the signed cast is
        exact. A repo-wide grep confirms no other instance (the two other `+= (...>>n)` sites add
        strictly non-negative terms). The file is #ifdef SPECTRAL_DEBUG_ARM (undefined in all five
        triad targets) so the fix emits no triad code change — verified green by construction (5 builds
        clean, ctest 4/4) AND separately proven to compile body-active via `clang -fsyntax-only
        -DSPECTRAL_DEBUG -D__ARM_ARCH_7EM__` (edited lines emit no diagnostic). The three psychoacoustic/
        sinusoidal proc stages are confirmed no-op stubs; processing-chain mask parse/dispatch (overflow-
        checked, none/saw_none-enforced, bounded snprintf advance), console formatting (clamped pads,
        bounded snprintf, filled in [0,width]), the log shim, and the libsndfile audio-in path (sf_count
        ->size_t representability proven before overflow-checked allocs, finite-validated, leak-free; the
        downmix base=i*channels < total_samples; audio_window clamps to [0,total_frames] with NaN end_sec
        caught downstream) are all clean.

  [188] Daisy Seed firmware glue + UART command protocol (api/daisy_seed/daisy_seed_spectral.c) —
        CLEAN AUDIT, no defect, no code change. The UART byte-stream command state machine (the one
        adversarial surface — bytes from an external host) is memory-safe: DATA-state fill checks
        `data_len >= MAX_MSG_LEN` before every store (uint8_t counter vs uint8_t buffer, no overrun);
        fixed-len commands reach CHECKSUM only at data_len>=expected_len so the 4-byte memcpys are
        always in-bounds; LOAD_FILE only executes via the stored '\0' so fname is always NUL-terminated;
        the `..`-traversal reject reads p[2] only when p[0]==p[1]=='.' (both non-NUL) so its max index is
        the NUL itself (no OOB). get_memory clamps total_used<=SDRAM_SIZE before the subtraction (no
        underflow); load_sd/load_buffer reject num_segments>capacity before the pool copy; ADC/param maps
        CLAMP before use; playback API is NULL-guarded passthrough to the pass-181 embedded core. ONE
        deferred observation (NOT changed): load_sd reads the .spq segments straight into the pool and
        commits without re-validation (load_buffer validates via spectral_arm32_load). Bounded to audio
        quality NOT memory safety — the runtime activation gate `num_active<MAX_ACTIVE` + prune (pass 181)
        drop malformed segments rather than overrun. Deferred because the file builds in NO host target
        (arm-none-eabi + FATFS headers absent; SD code is #ifdef DAISY_HAS_FATFS) and the validator is
        static (a real fix needs an API change to triad code or a temp buffer) — unverifiable here, so
        maintainer-directed alongside the GPU fade-tail item. Verified: no source changed -> Pass 187
        green preserved (and the file is in no triad target regardless).

  [189] Tree-wide defect-CLASS cross-cut + last inline-logic headers (runtime/spectral_perf_accounting.h,
        core/spectral_omp.h, plus class sweeps A/B/C over the whole tree) — CLEAN AUDIT, no defect, no
        code change. Having swept every logic-bearing .c file end-to-end (161-188), this pass does the
        orthogonal thing: a cross-cut of the three defect CLASSES most likely to harbour a latent twin of
        the Pass 187 bug. (A) unsigned subtraction feeding a shift — grep tree-wide finds only the three
        Pass-187 sites (now fixed) plus two `+= (...>>n)` sites that add strictly NON-negative terms
        (wavetable masked checksum byte, perf_model ceiling-divide) — no surviving twin. (B) integer
        division by a runtime denominator — every variable divisor is guarded (perf_embedded:362
        `if(blocks==0)blocks=1`, arm32:1138 `if(call_count==0)return 0`) or a nonzero compile-time
        constant (segment_pool block_size, gpu_tile tile_size); the rest are double division (IEEE inf,
        no trap). (C) memcpy/memmove with a computed byte count — every site routes its size through the
        overflow-checked `*_bytes`/`size_mul` helpers (audited pass 185) or has a preceding `>= sizeof`
        bound-check (cli_pipeline path copies). The two remaining inline-logic headers are clean:
        perf_accounting.h casts to uint64 before `+3 >> 2` and early-returns before its multiply;
        omp.h clamps effective threads to [1, MAX_THREADS]. Verified: no source changed -> Pass 188
        (== 187) green state preserved by construction.

  [190] Tree-wide defect-CLASS cross-cut round 2 — float→int out-of-range conversion / signed-left-shift
        UB / transcendental-domain NaN injection — CLEAN AUDIT, no defect, no code change. The next three
        UB/NaN classes a DSP kernel is most exposed to. (D) float/double→int conversion (C11 6.3.1.4:
        out-of-range is UB): every site finite-guards AND saturates/range-reduces before the cast —
        spectral_q15.h float_to_q15/q31 (isfinite + clamp [-1,1]), phase_rad_to_q15 (fmod->[0,1) with the
        +1.0f-rounds-to-1.0f edge handled), omega_to_q88 (clamp 255 -> *256 < 65536), wavetable lookup_f
        (phase reduced to [0,1), idx<SIZE), synth_internal start/length (finite + >=0 + <out_len + <=SIZE_MAX
        BEFORE cast), cli_pipeline out_len (stretch validated finite-positive + <=MAX_STRETCH at the CLI
        boundary, host-only). (E) left-shift UB: every `<<` in the tree has an UNSIGNED left operand (1u /
        uint32_t) and an in-range count — no signed-overflow, no over-width/runtime count. (F) transcendental
        domain: asin clamps to [-1,1] before asinf; fast_sqrt/fast_inv_sqrt/fast_peak_log guard `x>0 &&
        isfinite` internally; windows.c sqrtf/log operate on sum-of-squares (>=0) with a LOG_FLOOR and
        finite-checks at every stage; peak_estimator fast_sqrt args are magnitude-squared (>=0). Verified:
        no source changed -> Pass 189 green preserved (5 builds clean, ctest 4/4 re-run to confirm).

  [191] Tree-wide defect-CLASS cross-cut round 3 — strict-aliasing type-pun / signed-integer-overflow in
        arithmetic — CLEAN AUDIT, no defect, no code change. The final two UB classes for a mixed
        float/fixed-point kernel. (G) strict-aliasing (C11 6.5/7): every float<->bits reinterpret goes
        through a `union` (endian.h, fast_math.c) — well-defined; the only raw pointer-casts are to
        COMPATIBLE types ((float*)fftwf_complex* — fftwf_complex IS float[2], FFTW's documented access;
        (void**)&segs for posix_memalign), and peak_track.c:1229 already uses memcpy specifically to avoid
        a pun. No incompatible-type pun exists. (H) signed-int-overflow (C11 6.5/5, UB unlike unsigned
        wrap): the arm32 fade ramps `(int32_t)pos*fade_step` are bounded < Q15_MAX by construction
        (fade_step=Q15_MAX/fade_len, pos<fade_len); the one large product (da_q15*sample_offset) is widened
        to int64 BEFORE multiplying; phase/freq accumulators are uint32 modular-by-design; the Pass-187
        EWMA deltas are << INT32_MAX (domain-proven). Adjacent re-confirm: the fade divisor Q15_MAX/fade_len
        is safe — the activator clamps fade_len to >=1 at arm32:857-859 (documented at 665-668), the
        embedded analogue of Pass 189's division sweep. Verified: no source changed -> Pass 190 green
        preserved (triad just re-run green on this tree).

  [192] Concurrency / FP-determinism cross-cut of every OpenMP parallel region — CLEAN AUDIT, no defect,
        no code change. No data race / incorrect shared accumulation anywhere. max-reductions
        (analysis_fft:400, analysis_fused:111) are associative+commutative over non-negative-finite magsq
        -> order-independent/deterministic. CPU additive synth (synth_cpu:239) gives each thread a PRIVATE
        buffer over a DISJOINT segment range (race-free) and reduces cross-buffer per-sample in FIXED
        ascending order (synth_cpu:197) (scheduling-independent). Fused analysis (analysis_fused:151) +
        peak tracker (peak_track:886) use per-thread private scratch + per-frame-pair output slots keyed by
        pair index (deterministic order), an atomic last_error flag, and local counters reduced after the
        region; pretouch/merge (peak_track:1203/1221) + gpu_tile (146/227) write disjoint indices. ONE
        explicit NON-defect note: the parallel additive-synth output differs from single-thread by <=1
        ULP/sample (FP-add non-associativity under a different reduction grouping when n_parts changes) —
        correct additive sum within float epsilon (<-140 dBFS, inaudible), deterministic per fixed thread
        count, NOT in tension with the binary-byte-identity rule (that's about the compiled binary), and
        the CTest oracle exercises the integer-exact Q15 path. Recorded, not "fixed" (a fixed serial
        reduction tree would cost throughput for no audible gain — KISS). Verified: no source changed ->
        Pass 190 green preserved.

  [193] Memory-safety lifecycle cross-cut (use-after-free / double-free / leak-on-error / uninitialized
        read) at the highest-risk cleanup sites — CLEAN AUDIT, no defect, no code change. 141 free() sites
        tree-wide; the risky shapes (inline-free + converging fail/cleanup label; one buffer freed across
        many early returns) are all disarmed by correct idioms: cli.c `skip` is freed at :493 then nulled
        at :494 so the fail-label free is NULL-safe, and `eff_argv_heap` success-frees at :507 and returns
        before the label (single free per path); seg_cache's two endian-swap scratch buffers (:524 Segment*,
        :549 SegmentGpu*) are distinct block-scoped vars each freed once BEFORE any goto and out of scope at
        the label; `entries` is freed on ~14 mutually-exclusive single-owner early returns; result_free /
        segment_mt_free / segment_pool_free are symmetric one-shot teardowns; gpu_tile/mmap'd payloads are
        freed only under an `owns_*`/`capacity>0` ownership flag (borrowed cache views never free()'d).
        Cleanup aggregates are `{0}`-initialized so an early goto frees NULL not garbage. Verified: no source
        changed -> Pass 190 green preserved. *** PHASE C CONVERGED: file-by-file (161-188) + every static
        defect class cross-cut (189 underflow/div0/memcpy, 190 float→int/shift/transcendental, 191
        aliasing/signed-overflow, 192 concurrency-race/determinism, 193 heap-lifecycle) all clean. The only
        remaining verification is RUNTIME numerical/algorithmic DSP-math correctness — exactly Phase D's
        golden-vector harness, also the home of the two deferred observations. ***

REMAINING (not yet swept): core/analysis/synth + CLI/orchestration + embedded synth
fade + core dispatch + binary-deserialization/converter surface + host GPU-tile
concurrency + canonical oscillator math contract + host SIMD oscillator port +
file-I/O/CLI boundary cluster + peak frequency-estimation cluster + SpectralTracker
lifecycle/storage/reduction + STFT analysis FFT/orchestration cluster + CPU additive-
synthesis/wavetable/oscillator-math cluster + GPU synthesis dispatch/timbre-gate cluster
+ segment-cache persistence cluster + Q15 ARM32 fixed-point synthesis cluster + host
file-I/O layer + CUDA tile-parallel synth backend + segment-storage/Q15-LUT/CPU-fade/
resource-bridge cluster + support/utility math cluster (hash lifecycle, resource-path
canonicalization vs Python, alloc/overflow helpers, perf cost model, Q15 primitives,
error/peak-model) + synthesis-backend-dispatch/wavetable-bank cluster + ARM debug-instrumentation/
analysis-proc-stub/processing-chain/console/log/audio-in cluster + Daisy firmware-glue/UART-protocol
cluster swept end-to-end (161-188, with a real unsigned-EWMA underflow fixed in the ARM DWT/ITM debug
monitor at pass 187) AND now EIGHT defect CLASSES cross-cut tree-wide — (189) unsigned-underflow-shift /
integer div-by-zero / computed-size memcpy + (190) float→int out-of-range conversion / signed-left-shift
UB / transcendental-domain NaN injection + (191) strict-aliasing type-pun / signed-integer-overflow — all
clean, plus (192) the full OpenMP concurrency surface (race-free + reduction-determinism) + (193) heap
lifecycle (UAF/double-free/leak/uninit), plus the final
inline-logic
headers. All major compute AND support surfaces (CPU additive/wavetable, Q15 embedded,
Metal + CUDA GPU, persistence/file-I/O, the utility-math layer, backend dispatch + wavetable load/lookup,
the debug-instrumentation + optional-processing-chain surface, and the Daisy firmware glue + UART
command protocol) now audited. TWO bounded, unverifiable-on-host observations are recorded (deferred,
maintainer-directed: the GPU fade-tail-under-time-stretch non-monotonicity, and the Daisy SD `.spq`
load skipping segment re-validation — both memory-safe, both await the relevant toolchain/golden-vector
loop). Phase D IN PROGRESS: D0 (harness infra) done ahead of schedule + D1 (full/fused parity)
landed (12/12 ctest green, full==fused 0-ULP on 6 fixtures); D2 golden-vector oracle NEXT
(maintainer sign-off on frozen fixtures/tolerances), D3/D4/D5 pending.
```

---

## Phase D — Compiled harness + tooling feedback (Phase G)

Objective: the abstraction the maintainer wants for testing — a proper,
extensible compiled harness for the C kernel — plus the out-of-source generator
feedback loop, plus retiring the brittle string-grep tests and fixing CI.

Decisions locked: CTest + small C harness; golden vectors as the canonical
oracle.

### D0 — Harness infrastructure

```text
- enable_testing() + a minimal C assertion/runner; add_test() targets in CMake.
  Link the kernel; one executable per component (oscillator, window, q15,
  peak-estimator, arm32-sim, analysis).
- Python keeps only static/lint duties; behavioral truth moves to C + CTest.
```

### D1 — Full/fused parity harness

```text
- Implement FULL_FUSED_PARITY_HARNESS.md against
  analyze_audio_with_path_mode(... PATH_FULL / PATH_FUSED ...): deterministic
  fixtures, sort, per-field tolerances, nonzero on failure. This is Phase G.
```

### D2 — Golden-vector oracle (canonical)

```text
- Commit golden fixtures (silence, DC, impulse, exact/fractional-bin sine, chirp,
  two-tone, dense) with per-field tolerances. This committed spec is the oracle.
```

### D3 — LUT generator feedback loop (golden vectors validate both sides)

```text
- The committed golden vectors are canonical. BOTH are validated against them,
  neither privileged:
    C runtime LUT      (spectral_lut_init_sine in core/spectral_lut.c)
    out-of-source gen  (tools/spectral_tools/generators/lut_generator.py)
- Use generators/native_bridge.py so the harness can call the C path and diff
  against the generator output and the golden set in one place.
- Generalize the pattern to other generators (resource_hashes.py) and document it
  via the tools ADR (ADR-0001) so the subtrees/ feedback loop is canonical.
```

### D4 — Retire string-grep tests + fix CI

```text
- Migrate the 131 tests/core_math/*.py string-matchers to contract/behavioral
  tests; delete those that only memorialize stale names (forbidden by
  KERNEL_PATCHING_GUIDELINES §6 yet currently pervasive, e.g. pass129/pass133).
- Fix .github/workflows/c-cpp.yml: it runs on the non-existent "debian-latest"
  and calls make check / make distcheck, neither of which the (CMake) Makefile
  defines, so the suite never runs in CI. Point CI at ctest + the pytest lint set.
```

### D5 — Backfill regression coverage

```text
- Convert every Phase A/B/C finding into a compiled regression test under the
  new harness.
```

### Phase D status (pass 221)

```text
- D0 (harness infra): DONE ahead of schedule. enable_testing() (root CMakeLists.txt:12)
  + 11 compiled per-component CTest runners + 2 benches accreted across the
  oscillator/Q-type sub-campaign (passes 194-220). No new D0 work needed.
- D1 (full/fused parity): LANDED (PASS221). tests/core_contracts/test_full_fused_parity.c
  + cmake/targets/full-fused-parity-test.cmake. Drives analyze_audio_with_path_mode()
  under PATH_FULL vs PATH_FUSED over 6 deterministic fixtures, sorts by (start,omega,amp),
  compares per-field within FULL_FUSED_PARITY_HARNESS.md tolerances, nonzero on fail.
  Result: bit-identical (0 ULP) on all 6 fixtures. ctest 12/12 green; desktop builds clean.
  Supersedes the role of the test_core_pass119_*_spec.py string-matcher (its deletion = D4).
- D2 golden-vector oracle: NEXT. Wants maintainer sign-off on the fixture set + frozen
  tolerances (this defines the numerical contract).
- D3 LUT generator feedback loop, D4 retire ~131 string-grep py tests + fix CI
  (.github/workflows/c-cpp.yml broken: debian-latest + make check/distcheck undefined;
  HIGH blast radius — deletes tests + edits CI, wants explicit go-ahead), D5 backfill: pending.
```

### Phase D closure criteria

```text
- ctest is green and run by CI on a real runner.
- Full/fused parity harness compiled and passing within documented tolerances. [D1 DONE — PASS221]
- Golden-vector oracle validates the C LUT and the Python generator together.
- No test asserts on source substrings except deliberate dangerous-pattern lints.
```

> Cross-reference (additive, non-conflicting): the optimisation track in
> `docs/core_audit/OPTIMISATION_PLAN.md` consumes this phase's harness + golden
> oracle as its verification gate. It introduces no new test philosophy and does
> not alter Phase D scope, ordering, or closure criteria — the goldens here freeze
> the current numerical contract that the optimisation work is verified against.

---

## Phase E — Core port-layer separation (embedded vs host; device-agnostic)

Objective: stop interleaving embedded (low-level / fixed-point) and host (float)
implementations inside shared core files. Today ~22 core/analysis files carry
`#if SPECTRAL_EMBEDDED` / `SPECTRAL_ARM_M7` / emulator-guard branches (~70 guard
lines in core/ alone), and device-specific detail has leaked into the core —
`spectral_config.h` hardcodes STM32H7 memory sections (`.dtcm_data`,
`.itcm_text`, `.sdram_data`). Refactor to the industry port-layer pattern: a
device-agnostic core that distinguishes *execution profiles* (low-level/embedded
vs host) through build-selected implementation files behind shared interfaces —
not #ifdef soup, and never device names.

Constraint (maintainer): the core MUST distinguish embedded/low-level from host
(legitimate and required), but stays completely device-agnostic — no specific
MCU/board anywhere in core. This is the faithful realization of the original
"refactor arches into separate files" item (the Pass-140 note only addressed SIMD
portability via SIMDe, a *capability* concern, not the embedded/host split).

### Base the design on these (how others solve this exact problem)

```text
- FreeRTOS: a fixed interface (portable.h / portmacro.h) + per-port implementations
  under portable/<toolchain>/<arch>/port.c, build-selected. The kernel never
  #ifdefs the port; board/device specifics live in the BSP, not the kernel.
- Linux: arch/<arch>/ implementations + include/asm-generic/ fallbacks, config-
  selected; generic code includes <asm/...> abstractions, never arch specifics.
- CMSIS: CMSIS-Core is device-AGNOSTIC (Cortex-M class); capability via feature
  macros (__ARM_FEATURE_DSP); concrete MCUs ship in separate Device Family Packs.
  This is precisely "distinguish low-level, stay device-agnostic".
- SQLite: the OS layer is an interface (sqlite3_vfs) with os_unix.c / os_win.c
  backends; the SQL core carries no platform #ifdefs.
- musl libc: generic C with per-arch overrides under src/<subsys>/<arch>/.
```

### Target architecture for this repo

```text
- core/ = portable, profile-agnostic kernel: algorithms, contracts, interfaces.
  No SPECTRAL_EMBEDDED / ARM_M7 branching in logic bodies; it depends on abstract
  profile interfaces (the "what").
- Per-profile implementations in build-selected files (the "how"), e.g.
  core/port/host/<x>.c and core/port/embedded/<x>.c behind one shared header.
  CMake targets pick the source set — exactly as synth/backends already does —
  replacing in-file #if branches with file selection.
- Capability gates (SIMD / DSP / FPU presence) stay feature-macro based (SIMDe on
  host, CMSIS/feature on embedded). Capability is not device, so it stays portable.
- DEVICE specifics leave core entirely. The STM32H7 memory-section macros become a
  device-AGNOSTIC memory-class abstraction the core uses by intent
  ("fast/tightly-coupled" vs "bulk/external"); the concrete binding (.dtcm_data,
  linker script, DMA, cache maintenance) lives in the BSP/port under api/. Core
  asks for a memory class; the board provides it. (Pass 142 gated the section
  attrs to real ARM; Phase E moves the binding out of core.)
```

### Sequencing

```text
- Needs the verification foundation (A1b harness / Phase D seam) so each file
  split is provably behavior-preserving (moves are byte- or tolerance-identical,
  guarded by the harness/oracle).
- Best before the deep ARM hot-loop redesign (A2/A3): a clean per-profile layout
  makes that redesign localized instead of surgery through #ifdefs.
- Large and cross-cutting; one module's split = one pass (move + verify), not a
  big-bang. Final order at maintainer's direction.
```

### Phase E closure criteria

```text
- No SPECTRAL_EMBEDDED / SPECTRAL_ARM_M7 branching inside core algorithm bodies;
  profile differences are build-selected implementation files behind shared headers.
- No device name or device-specific section/peripheral detail anywhere in core;
  memory-class and other device bindings are provided by the BSP/api.
- Capability (SIMD/DSP/FPU) remains feature-detected and device-agnostic.
- Every module split verified behavior-preserving by the harness/oracle.
```

### Phase E status (passes 147–155)

```text
Core port-layer separation is complete; the verifiable closure criteria are met.

DONE
- Criterion 1 (no profile/device branching in core algorithm bodies): met. The
  embedded/host-divergent kernels were extracted into build-selected port files
  behind shared, unconditional headers:
    pass 152  oscillator_simd      -> core/port/{host,embedded}/oscillator_simd.c
    pass 153  spectral_vector_ops  -> core/port/host/spectral_vector_ops.c
    pass 154  spectral_out kernels -> core/port/{host,embedded}/spectral_out_kernels.c
    pass 155  gpu_tile_preprocess  -> core/port/{host,embedded}/spectral_gpu_tile.c
  No SPECTRAL_EMBEDDED / SPECTRAL_ARM_M7 / SPECTRAL_RESTRICTED_MODE branch remains
  inside any hand-written core .c algorithm body (only the generated resource-hash
  table and the per-profile port files retain capability/profile selection, which
  the criteria permit). Profile selection that stays is confined to the canonical
  config/macros headers (e.g. SPECTRAL_FADE_SAMPLES_ACTIVE) and to capability- or
  filesystem-driven type/layout choices in headers (spectral_resource_fs.h struct,
  SPECTRAL_FORCEINLINE), not to algorithm logic.
- Criterion 3 (capability feature-detected): met. OSC_SIMD_*, SPECTRAL_USE_CMSIS,
  SPECTRAL_USE_VDSP, __ARM_FEATURE_DSP, SPECTRAL_HAS_FILE_IO are all feature macros.
- Criterion 4 (each split behavior-preserving): met. Every pass verified against
  the same triad — six green targets build, ctest arm32_process_correctness passes,
  the sim oracle matches all 6 goldens, and the desktop render is byte-identical to
  the pre-refactor binary (cmp-clean).
- Pass 151 made the wavetable sample type a profile-selected spectral_sample_t
  abstraction; pass 147 introduced the device-agnostic memory-class macros
  (SPECTRAL_MEM_FAST / _FAST_CODE / _BULK, core/port/spectral_mem.h).

OPEN (criterion 2 residual)
- The concrete Cortex-M section-name binding (.dtcm_data / .itcm_text / .sdram_data)
  still lives as a built-in default in core/port/spectral_mem.h. It is already
  isolated to the port layer (no device detail in any algorithm file), abstracted
  behind the memory-class intent macros, and overridable by a board via
  SPECTRAL_BSP_MEM_HEADER; the default is gated to a real Cortex-M cross-compile
  (__ARM_ARCH_7EM__), so it is inert on all six green targets.
- Relocating that default into the api/ BSP (e.g. api/daisy_seed) is the final
  step. It is deliberately deferred: the only consumer is the bare-metal daisy
  firmware build (arm-none-eabi), which is NOT in the green target set and cannot
  be exercised here, so removing the core default would be an unverifiable change
  to firmware memory placement. Do it together with an embedded-toolchain build so
  the move is verifiable rather than preprocessor-reasoned.
```

---

## Cross-phase done definition

```text
- ARM path exploits the architecture with benchmark + parity evidence.
- Every breakable guarantee is registered, budgeted, and self-reported.
- Kernel is swept clean of embarrassing-class defects, each pinned by a test.
- A compiled, extensible harness is the source of behavioral truth, with the
  out-of-source generators validated by canonical in-repo golden vectors.
- The core is device-agnostic: embedded vs host differences are build-selected
  implementation files behind shared interfaces; device specifics live in the BSP.
```
