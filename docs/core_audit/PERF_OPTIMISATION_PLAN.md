# Performance optimisation plan — profile-driven, algorithm-first, then asm-tune

Goal: optimise toward the assembly level, but only where the algorithm is already the right one
(no "house on sand"). This plan is grounded in real profiles on both targets, not guesses.

## Part 0 — Perf-system audit (measure clean before you optimise)

You can't profile-drive optimisation off a noisy or inconsistent meter. Audited the desktop
benchmark harness (`tools/spectral_tools/testing/benchmark_runner.py`, `benchmark_parsing.py`)
and the ARM/M7 model (`benchmark_workflow m7-*`, `tests/tools/test_perf_gate.py`). Findings + fixes:

- **Cold-run contamination of the per-stage breakdown (fixed).** The headline `Total ms` and
  `Memory max RSS` lines reported `warm_*` (cold first run discarded), but the per-stage medians,
  bandwidth medians, and cache medians folded the cold run *in*. The cold run is a real ~2× outlier
  on the FFT stage (measured: fft run-01 ≈ 77 ms vs warm ≈ 40 ms; track 17→11; norm 4.3→1.5 — the
  one-off FFT/vDSP plan build, first-touch paging, dyld bind). Now every per-stage aggregate is
  reported over the **warm set** (`_warm()` drops run 1 when runs>1), so the breakdown reconciles
  with the headline `warm_median` instead of silently including the outlier. Lines relabelled
  `Stage warm medians ms:` / `Stage bandwidth warm medians:` / `Segment-binary warm medians ms:`.
- **Run-to-run jitter was invisible (fixed).** Only `track` exposed spread; real jitter on synth
  (measured 6.7→15.7 ms within one sweep) and norm (1.0→4.3 ms) hid behind a bare median, so you
  couldn't tell a trustworthy number from the midpoint of a 2× swing. Added a
  `Stage warm spread ms (min..max):` line + a parsed `stage_spread` metrics section.
- **Parser overrun bug (fixed).** The summary regexes captured the last token on each line with
  `([^ ]+)`; `[^ ]` matches newlines, so the trailing group swallowed `norm=1.6\nStage…` and
  failed the float parse — `norm_ms` and `warm_mean_ms` silently came back `null`. Latent only
  because nothing consumed those keys. Switched the captures to `\S+`. Pinned by
  `tests/tools/test_benchmark_parsing.py` (overrun regression + warm/spread/legacy parsing).
- **RSS path verified correct.** macOS `/usr/bin/time -l -o <file>` supports `-o` and emits
  `maximum resident set size` in bytes; the BSD branch divides by 1024 correctly. No change.
- **ARM/M7 model verified.** `test_perf_gate.py` + `test_embedded_perf.py` = 39/39 green, nothing
  skipped: the live qemu-counts / mca-validation / WCET stack runs and sits within the frozen
  `m7_baseline.json` contract. No change needed.

Pre-existing, out of scope (flagged separately): `test_layering.py` include-direction violation
`spectral_synth_ifft.c [kernel] -> spectral_fast_sin_simd.inc [arch/simd]` (from the IFFT
shared-SSOT extraction, commit 42bc3c2baf) — an IFFT-workstream layering decision, not perf tooling.

## Part 1 — Assembly as string / generated code → files

**Finding: there is essentially no CPU-assembly-as-string to extract.** Exhaustive sweep:
- The only inline asm is single-instruction ARM memory **barriers** — `dsb`/`isb` in
  `arch/arm/spectral_synth_arm32.c` and `spectral_debug_embedded_arm.c`. These **must stay inline**
  (a barrier in a separate `.S` function defeats its purpose; the compiler cannot order around it
  otherwise). Do NOT move them.
- `spectral_smlald` (`core/spectral_q.h`) is the `__smlald` **CMSIS intrinsic** (dual 16×16 MAC),
  with a portable C fallback — correctly an intrinsic, not asm-as-string.
- The only generated code embedded as a C string is the **Metal MSL shader**
  (`drivers/metal/spectral_osc_metal_generated.h`, codegen'd from the C SSOT by `metal_osc.py` and
  verified by `verify_metal_osc`). That is GPU code, not CPU asm.

**Recommendation:** keep barriers inline. When we add hand-tuned CPU asm (Part 2 §3), put it in
`arch/<isa>/*.S` files — the `arch/{arm,simd,ref}` layout already segregates by ISA, so the home
exists (see `archive/ARCH_PATH_SELECTION.md`). The MSL is GPU; moving it to `.metal` files is
possible but would replace the SSOT-verified codegen-as-string with a runtime-loaded file — only
worth it if/when GPU-shader perf work needs hand editing. No action needed today.

## Part 2 — Profiles

### Mac / desktop (host: analysis + synthesis) — `sample` on shakespeare.wav

**Stage timing + threshold sweep** (n_fft=4096, hop=128) — the key algorithmic-stability result:

| db_thresh | segments | Track | **FFT** | dominant stage |
|---|---|---|---|---|
| -85 (bench stress) | 4.25M | 73% | 27% | Track |
| -60 | 716K | 45% | 55% | mixed |
| -40 (realistic) | 180K | 21% | **79%** | **FFT/STFT** |
| -20 | 18K | 7% | **93%** | **FFT/STFT** |

Track time is **linear in segment count** (~70–90 ns/segment across the sweep) → the tracker is
algorithmically stable; the 4.25M-segment / 99%-acceptance regime is purely a too-low-threshold
*parameter* artifact, not an algorithm defect. **At realistic thresholds the STFT/FFT stage
dominates.**

**Function-level leaf hot spots** (`sample`, CPU-synth forced):
1. ~~`isfinite` (`__isfinitef`/`__isfinited` libm stubs) — ~2300 samples, the #1 cost~~ → **FIXED**
   (commit: inlined bit-trick; **−85% → ~340 samples**; `spectral_fft_single_frame` 974→222).
2. `spectral_osc_simd_segment_sine` — ~1280 samples, now the #1 leaf (the SIMD minimax-sine synth).
3. `VVATAN2F` — ~410 samples (the all-bins phase atan2).
4. `spectral_fft_single_frame` — ~222 (post-isfinite-fix; the STFT transform + per-bin magsq).
5. OMP outlined overhead — a few hundred (parallel granularity).

### ARM / embedded (Daisy M7: synthesis only; analysis is host-side)

Profiled via the **m7 cycle model** (`tests/fixtures/m7_baseline.json` — `worst_cyc_per_voice_sample`,
`kernels`, `wcet_scenarios`; the arm32 synth codegen is byte-pinned). Hot path = the Q15 oscillator
inner loop `synth_core_m7` (scalar Q15, unrolled ×4) and the dual-16-bit-MAC sustain path
(`spectral_smlald`, `SPECTRAL_HAS_DUAL_MAC`). The limit is per-sample Q15 MAC throughput.

## Algorithmic-stability assessment (don't build on sand)

| Hot spot | Algorithm settled? | Verdict |
|---|---|---|
| Peak tracker (track stage) | Yes — linear in segments, MQ matching | STABLE. Optimise per-segment emit only if a low-threshold use case demands it. |
| STFT transform | Yes — vDSP/FFTW (library, near-optimal) | Not ours to hand-tune; tune FFTW planning (below). |
| Per-bin phase `atan2` (all bins) | **NO — computing phase for all bins is wasteful** | **Settle first:** the phase-at-peaks refactor (compute atan2 at peak bins only). |
| SIMD minimax sine (synth) | Yes — minimax fold is SOTA for this | STABLE → asm-tune candidate. |
| Q15 oscillator MAC (ARM) | Yes — cycle-modeled, dual-MAC | STABLE → ILP/scheduling tune within the perf gate. |
| `isfinite` | n/a (not an algorithm) | DONE (inlined). |

## Prioritised roadmap (algorithm-first, then asm-tune the settled kernels)

1. **DONE — `isfinite` inline bit-trick** (−85% of the isfinite cost, measured, ctest 32/32).
   *Follow-up (small):* sweep the remaining raw `isfinite()` in `spectral_peak_estimator.c` (45
   sites, f32/f64 per site) and `spectral_peak_track.c`/`_interp.c` through the helper.
2. **ALGORITHM — phase-at-peaks refactor** (retires the all-bins `atan2`: `VVATAN2F` + a chunk of
   `spectral_fft_single_frame`). Fully designed in `ANALYSIS_PHASE_AT_PEAKS_PLAN.md`. Land this
   BEFORE asm-tuning the FFT stage — it removes most of the per-bin work, so tuning the remaining
   bins would be wasted otherwise.
3. **ASM-TUNE the settled kernels** (only after 1–2):
   - `spectral_osc_simd_segment_sine` (now the #1 leaf): the width-templated SIMD kernel
     (`arch/simd/spectral_oscillator_simd_kernel.inc` + `spectral_fast_sin_simd.inc`). Run
     `llvm-mca` on the inner loop to find the ILP/port bottleneck; apply FMA contraction,
     `__restrict`/alignment hints, and unroll/interleave tuning; only drop to hand-written `.S`
     (in `arch/simd/`) if the intrinsics can't reach the mca-predicted throughput.
   - ARM Q15 oscillator (`synth_core_m7`): per the m7 model, push dual-MAC coverage; any change is
     gated by the m7-baseline (regenerate deliberately). `.S` candidate if the C can't hit the
     cycle target.
   - The per-bin magsq/phase SIMD (`spectral_vector_ops.c`): verify mca-optimality post-phase-at-peaks.
4. **Build-system wins (free, no asm):**
   - **PGO** — the engine already has `-fprofile-generate/-use` wiring (`host-config.cmake`).
     Generate a profile from the bench workload, rebuild with `-fprofile-use`. Typically 5–15% on
     branchy code like the tracker for near-zero effort.
   - **FFTW planning** on Linux — `FFTW_ESTIMATE` (current) → `FFTW_MEASURE` (one-time slower plan,
     faster execution) for long offline analyses; measure the tradeoff.
5. **Tooling:** add an `llvm-mca` / instruction-level harness to `benchmark_workflow` for the
   hand-tuned kernels, so asm-level changes are measured against a port-model prediction, not vibes.

## Measured outcomes (update)

- **isfinite (item 1): DONE — the headline win.** Inlined bit-trick + `_Generic` sweep across the
  FFT-stage per-bin path, the estimator, the tracker, and the contracts. `sample` shows the
  analysis-side leaves collapsed (`spectral_fft_single_frame` 974→14, `isfinite` stubs gone). The
  profile is now synth-dominated.
- **Synth-sine asm-tune (item 3): DECLINED on data (M-series).** `llvm-mca` calls the inner loop
  latency-bound (18.8 cyc/iter vs 10.7 throughput, IPC 1.92/6) and the compiler doesn't unroll it.
  But forcing `#pragma clang loop unroll_count(2)` **regressed synth ~1.8×** (18–24ms → 32–55ms):
  doubling the minimax-sine's live vector temps spills registers, and the M-series' deep OoO window
  already overlaps the independent iterations. The math is `osc_parity`-pinned (no Estrin/reorder).
  Conclusion: the kernel is near-optimal on this target. **Re-try the unroll on x86/AVX2** (16 ymm
  regs may absorb the pressure) — un-measurable on this ARM box, so a CI item, not a local change.
- **Phase-at-peaks (item 2): re-scoped DOWN.** After the isfinite win the FFT stage shrank, so the
  all-bins `atan2` is a much smaller share than when first scoped. Remaining value is accuracy
  (exact phase on every platform; drops the per-peak reconstruct round-trip), not speed. It is a
  ~20-edit atomic refactor at +50% full-path memory. Worth doing only if the accuracy/cleanliness is
  wanted; gate it on a real before/after measurement, and if landed make it the unconditional
  default (analysis is host-only, so there is no memory-constrained target to flag it off for).

## Why this order

The profile says: at realistic settings the **STFT stage dominates**, and within it the biggest
*ours-to-fix* costs were `isfinite` (done) and the all-bins `atan2` (algorithmic — phase-at-peaks).
Only once those are settled does hand-tuning the SIMD sine and Q15 MAC pay off; doing asm work on
code that's about to be restructured (the atan2 path) or on a check that should just be inlined
(isfinite) would have been the "house on sand."
