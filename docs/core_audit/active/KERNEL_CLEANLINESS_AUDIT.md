# Kernel Cleanliness Audit — the maintainer's 10-concern sweep

**Status:** LEDGER OPEN. Branch `embedded-arch-audit`. Read-only audit complete; execution
gated per-wave below. **Lane:** desktop / core / host + cross-cutting architecture. The
embedded determinism/perf stack (`DETERMINISM_SURFACE_PLAN.md`: `spectral_debug_embedded_arm.*`,
the perf tooling, device profiles, `m7_baseline.json`, the perf-gated `arch/arm/spectral_synth_arm32.c`)
is owned by a parallel agent — findings there are marked **propose-not-prescribe**.

Method: 11 parallel read-only finders (config/macros, flag-taxonomy, layering, device-placement,
duplication, comments, dead-wiring, generated-files, legacy-defaults, K&R deep-read, tests-docs)
→ 105 raw findings → dedup/rank/sequence. Highest-value Wave-1 claims hand-verified before
persisting (forbidden estimator, dead-symbol ref counts, fade dup, `OPT_LEVEL`).

## Progress

**Maintainer decisions (2026-06-21):** keep `arch/` sibling + invert the leaks (#3); keep borderline
forward surface, delete scaffolding (#10); draft `FLAG_TAXONOMY.md` for review, no renames yet (#4).

**Wave 1 — dead-code sweep: DONE + verified.** Deleted: the native-synth twins subsystem
(`synth_cpu_native`/`_wavetable_native` + `synth_preflight_native` + `thread_buffers_combine_native`
+ `reduce_native_wrapper` + the 2 native segment callbacks + `NativeWavetableCtx`), `spectral_audio_write_stereo`
+ `spectral_mono_to_stereo_float`, the wavetable trio (`_lookup_q`/`_save`/`_generate_builtins` + the
now-orphan `_builtin_sample`), `_window_descriptor_at`/`_find_by_id`, `_hash_file_method_descriptor_count`/
`_descriptors`, `_ifft_synth_n_fft`, `_fs_write`, `_getenv_f64`/`_f64_positive`, `SPECTRAL_OPT_LEVEL`
(+ guarantees.h tombstone), `OSC_SIMD_WIDTH`, the forbidden `p*=1.5f` estimator (`SPECTRAL_TRACK_INTERP_POWER_RATIONAL`).
KEPT as forward surface: `_load_buffer`, `renderer_by_id`, the backend query trio, peak-policy stringifiers.
**Consequence handled:** deleting the native twins left the float-OpenMP synth machinery
(`synth_cpu_driver` + `thread_buffers_*` + `synth_partition_count`) with no caller under
`SPECTRAL_USE_EMBEDDED_SYNTH` → wrapped it in the existing `#ifndef SPECTRAL_USE_EMBEDDED_SYNTH`
guard (embedded synth is arm32-owned). **Verified:** desktop + simulate + embedded_arm build clean
(0 warnings); `tests_all` builds; ctest all wired tests pass.

**Wave 1 — remaining (next):** comment scrubs (strip fragile perf numbers + plan-phase labels, ~14
files), trivial SSOT/config hygiene (`dup-q31-pi-halfpi`, `dup-sine-lut-builder`, `dup-bandlimited-realloc`,
`config-unguarded-tunables`, `config-fade-samples-no-budget`, `config-peak-policy-downstream-enum`,
`config-sample-span-finite-macro`), `renderer-subtractive-lie`, `over-exposed-helpers` (make static).

**Observation for the embedded agent:** `arm_admission_cap_test` is `EXCLUDE_FROM_ALL` and NOT a
dependency of `tests_all`, so a plain `ctest` reports `arm32_admission_cap` as **Not Run** (AI.md #2
wiring / #12 silently-skipped). It passes when built explicitly. Add it to the `tests_all` aggregate.

---

## Verdict per concern

| # | Concern | Verdict | Where |
|---|---|---|---|
| 1 | Macros in config.h | **Real, moderate.** 750-line, 138-define grab-bag spanning ~8 domains vs its self-declared 5; one function-like macro that should be a `static inline`; Q15 saturating-arith bodies; ~24 diagnostic-string macros; 2 unguarded tunables among ~100 guarded. | `core/spectral_config.h` |
| 2 | hash table `.c`→`.h` | **Instinct is C-incorrect** (external-linkage defs ⇒ `.c`; `.h` would multiply-define) **but points at a real smell**: the generated table embeds an `#if SPECTRAL_EMBEDDED` struct-shape switch, a hand-shadow stub, no digest, opaque `xx32_xx3` name, under-documented convention. | `core/spectral_hash_resources_xx32_xx3.c` |
| 3 | arch under core | **Recommend REJECT the nest.** The dependency-DAG law rests on the sibling kernel/arch/drivers distinction; nesting reads as "core may contain ISA bodies" — the opposite of #7. The real fix is #7's dependency inversion. | — |
| 4 | Flag naming (HEADLINE) | **Bad.** ~25 gating flags, no axis rule (`USE_`/`IS_`/`HAS_`/`_MODE`/bare mixed on one axis), 3 embedded-sim synonyms, 4 "restricted" spellings, 2 approx gates outside `ENABLE_APPROX_`, no SSOT doc, no test, AI.md's list rots (names 6 of ~25). | `core/spectral_config.h`, `AI.md:67` |
| 5 | Fragile perf comments | **Widespread.** Baked cycle counts / dBFS floors / ns-per-sample / speedup ratios (several duplicating the test SSOT) + a large plan-phase-label class (F1/F2b/Stage-N/Step-N/B1) leaked as planning narrative. | ~14 files |
| 6 | All vDSP in drivers? | **No.** The forward real-FFT (vDSP + FFTW3f) lives entirely in `analysis/`, with `FFTSetup`/`vDSP_Length`/`fftwf_plan` leaking into a shared struct. Only the IFFT reached `drivers/vdsp/`. | `analysis/spectral_analysis_fft.c` |
| 7 | core = control surface | **Violated.** `spectral_oscillator_dispatch.h` unconditionally `#include`s `arm_math.h`/SIMDe into the kernel and transitively into the Metal/CUDA drivers + CPU synth; `phase_nco8.h` and `synth_ifft.c` carry raw SIMDe bodies. `test_layering.py` is blind to it (ignores third-party includes). | core/ headers |
| 8 | Redeclared consts/funcs | **Several real.** Fade formula dup'd byte-for-byte (no parity test); Q15-timbre set hand-listed 3×; `osc_q31.h` defines `HALF_PI` twice + local PI bypassing consts.h; sine-LUT builder copy-pasted. | — |
| 9 | Legacy / 512 default | **512 storage default is correctly wired.** Live defects: `DAISY_MAX_ACTIVE=128` dead legacy macro contradicting it; the forbidden `p*=1.5f` estimator kept as a no-target branch. | — |
| 10 | Wiring / API surface | **Substantial dead surface.** ~14 zero-caller exported functions, 3 dead config macros, a dead alias that is (wrongly) the one pinned in the freeze test, several over-exposed internal helpers. | — |

**Bottom line:** no correctness blockers. Two genuine structural defects (#7 control-surface
leak, #6 partial FFT migration), one large naming debt (#4), and a big mechanical-cleanup
surface (#5, #10, #8). The kernel is in strong shape; this is hardening for upstream.

---

## Execution waves

### Wave 1 — Safe mechanical sweep (gated on: nothing)
Zero structural risk, no collision, each is a delete or comment edit traceable to one finding.

**Dead code (verified def+decl-only):**
- `dead-native-synth-twins` — `synth_cpu_native` / `synth_cpu_wavetable_native` + orphan helpers (`synth_preflight_native`, `thread_buffers_combine_native`, `segment_fn_native_*`, `reduce_native_wrapper`); `kernel.h:18` "+ native twins" surface claim is false. ✓verified 0 refs.
- `dead-backend-queries` — `spectral_backend_max_timbre` / `_supports_wavetable` / `_available`. ✓2 refs each.
- `dead-mono-stereo-write` — `spectral_audio_write_stereo` + orphan `spectral_mono_to_stereo_float`. ✓
- `dead-wavetable-fns` — `spectral_wavetable_lookup_q` / `_save` / `_generate_builtins` (+ `_load_buffer`: see API policy).
- `dead-peak-policy-stringifiers` — wire into the AI_CANON-#29 estimator/policy log **or** delete (wiring preferred).
- `dead-window-accessors` — `spectral_window_descriptor_at` / `_find_by_id`. `dead-hash-enum-accessors` — `_method_descriptor_count` / `_descriptors`. `dead-ifft-getter` — `spectral_ifft_synth_n_fft`. `dead-fs-write` — `spectral_fs_write`. `dead-getenv-f64` — `_f64` / `_f64_positive`.
- `dead-opt-level` — `SPECTRAL_OPT_LEVEL` + its guarantees.h tombstone. ✓ `dead-osc-simd-width` — `OSC_SIMD_WIDTH` (AI_CANON #30). `forbidden-rational-estimator` — drop the `p*=1.5f` branch + macro. ✓
- `over-exposed-helpers` — make `spectral_window_sum/_energy`, `spectral_log_level_name`, `spectral_getenv_nonempty` static.

**Comment scrubs (strip rotting numbers / plan-phase narrative; keep the WHY):**
`comment-osc-q31-cyc`, `comment-ifft-dbfs`, `comment-plan-phase-labels` (largest class: F1/F2b/Stage-N/Step-N/B1), `comment-oscillator-speedup-cluster`, `comment-phase-nco8-ratio`, `comment-perf-c-history`, `comment-analysis-old-layout`, `comment-windows-what` (keep sidelobe dB + Harris citation), `comment-gpu-seg-history`, `comment-q-c-30pct`, `comment-bandlimited-ulp`, `gen-c-stale-comment`, `comment-arm32-retired` (perf-gate: header block only, implementer confirms no codegen move).

**Trivial SSOT / config hygiene:**
`dup-q31-pi-halfpi`, `dup-sine-lut-builder`, `dup-bandlimited-realloc`, `config-unguarded-tunables`, `config-fade-samples-no-budget`, `config-peak-policy-downstream-enum`, `config-sample-span-finite-macro` (macro→`static inline`).

**Report honesty:** `renderer-subtractive-lie` (log the path actually run, not "subtractive" — `filter_envelope.c` has zero non-test callers), `backend-effective-timbre-stale` (low-confidence; verify against Metal/CUDA TLS first).

### Wave 2 — Bounded SSOT extraction + arch-body placement (gated on: nothing)
`dup-fade-envelope` (+parity test), `dup-q15-timbre-3x` (X-macro), `dup-wavetable-switch`, `dup-phase-to-rads-forwarder` (perf-gate on cmsis), `config-resolution-strings` (→runtime diagnostics header), `out-kernels-wrong-dir` (`arch/ref/spectral_out_kernels.c` → `arch/arm/`), `synth-internal-gpu-grabbag` (split GPU plumbing out), `phase-nco8-core` (→`arch/simd/`), `dead-q15-seg-size`.

### Wave 3 — Control-surface header split (the #7 unlock) — gated on: maintainer
`disp-isa-leak` (split `spectral_oscillator_dispatch.h` into a pure CONTRACT header in core + ISA selection in the arch TUs), `q15-eval-table-core` (perf-gate), `ifft-simd-core` (hoist the SIMDe trig batch behind an arch contract), **`layering-test-isa-blind`** (the new `test_core_no_isa_bodies` gate must land alongside), `ai-md-layering-overstated` (doc caveat).

### Wave 4 — Generated-file convention + config.h decomposition — gated on: maintainer
`gen-data-preproc-struct-switch` (designated initializers vs struct-shape `#if`), `gen-no-content-digest`, `gen-bridge-stub-dup`, `gen-opaque-xx32-name`, `no-generated-suffix-test`, `config-sample-abi-bodies` (→`spectral_sample.h`?), `config-grabbag-split` (large), `dup-hybrid-nfft-consts`. Tracked-accept: `config-gpu-tile-speculative`, `precise-phase-unbuilt`, `q31-kernel-in-core-keep`, `ifft-init-uncalled-tracked`, `hybrid-fade-divergence`.

### Wave 5 — Large architectural calls — gated on: maintainer
`fft-vdsp-in-analysis` (extract forward FFT into `drivers/vdsp/` behind a `SpectralFftBackend` port; per-thread FFT-setup pooling is the design risk), the flag-taxonomy rename cluster (`flag-no-ssot`, `flag-prefix-per-axis`, `flag-embedded-sim-triplet`, `flag-host-guard-divergence`, `no-flag-taxonomy-test`), `daisy-max-active-dead` / `daisy-max-segments-unused` / `dead-opt-level-dup` / `dormant-dma-branch` (embedded-agent), `comment-config-voice-ceiling` (embedded-agent).

---

## Durable tests to add (concern-class gates)
1. **`test_core_no_isa_bodies`** — fail when a `core/`/`runtime/` TU includes an ISA basename (`arm_math.h`, `simde/*`, `arm_neon.h`) or references raw intrinsic symbols (`simde_`, `__m128`, `_mm_`, `vld1q`, `vDSP_`, `arm_*_q31`), with a named allowlist for the one sanctioned `.inc`. **Closes the #7 gap `test_layering.py` is blind to.** Keys on symbol/include *classes*, not a file list.
2. **`test_device_api_confinement`** — vDSP/Metal/CUDA/CMSIS symbols originate only from `drivers/`+`arch/`, never `analysis/`/`core/` (nm on objects with a static-source fallback). Closes #6.
3. **`test_build_flag_taxonomy`** — every consumed `SPECTRAL_*` flag matches exactly one axis prefix and there are no N-synonyms-for-one-concept. Gated on the blessed taxonomy.
4. **`test_fade_envelope_parity`** — host `envelope.c` fade ≡ GPU `osc_formulas.h` fade over a sweep (degrades to one-definition check after dedup).
5. **`test_q15_timbre_set_single_source`** — the Q15 timbre set agrees across the 3 predicates.

## Docs to update (in the same patchsets)
- **AI.md:66-69** — replace the partial flag list with a pointer to a new `reference/FLAG_TAXONOMY.md`.
- **AI.md:32-34** — caveat that `test_layering.py` enforces include *direction* only and is blind to third-party/ISA includes; ISA-freedom in core is a separate gate.
- **reference/FLAG_TAXONOMY.md** (new) — the prefix-per-axis rule + full inventory + old→new map (after the scheme is blessed).
- **reference/GENERATED_ARTIFACTS.md** — state the `.c`-vs-`.h` extension-by-linkage rule explicitly; reconcile "digest-stamped" vs "verified by regenerate-and-diff".
- **AI.md:74 + `osc_formulas.h:7`** — correct the false fade SSOT claim.
- **`config.h:692` vs `:709`** — reconcile the `DAISY_MAX_ACTIVE` "real-time cap" vs "stale legacy" contradiction.

## Open decisions (maintainer-only)
1. **arch-under-core (#3):** recommend REJECT-nest; do the dependency inversion instead. Confirm keep-sibling.
2. **Flag scheme (#4):** bless axis prefixes (proposed `SPECTRAL_TARGET_` / `MODE_` / `HAS_` / `APPROX_` / `DEBUG_`); fold `USE_VDSP/CUDA/CMSIS`→`HAS_`, `METAL_FAST_MATH/PRECISE_PHASE`→`APPROX_`, collapse the 3 embedded-sim flags. Partly mechanical, partly embedded-agent-owned semantic collapse.
3. **Hash file (#2):** keep `.c` (recommended); decide designated-initializers + the `xx32_xx3`→`spectral_resource_hashes` rename.
4. **512 (#9):** delete `DAISY_MAX_ACTIVE=128`; embedded-agent decides the runtime admission cap.
5. **FFT-driver extraction (#6):** size the forward-FFT port move.
6. **`PRECISE_PHASE` / `EMBEDDED_FLOAT`:** wire+golden or delete the unbuilt branch + byte-identical reserved target.
7. **Dead-API policy:** delete all verified-dead (aggressive, pre-1.0 upstream-tightening) vs keep borderline forward surface (`_load_buffer`, `renderer_by_id`). Changing the freeze-pinned `spectral_synth_dispatch` alias is a deliberate API call.

## What to double-check before acting (honesty)
- `backend-effective-timbre-stale`: Metal/CUDA synth bodies were NOT read — verify the TLS-mutation-before-error assumption.
- Dead-symbol reach via X-macro/token-paste/function-pointer tables/Python-ctypes-by-string not exhaustively ruled out (12 confirmed at 2 refs; driver-only callers easy to miss).
- `_load_buffer` / `renderer_by_id` may be deliberate pre-1.0 longevity surface — lower-confidence deletes.
- `runtime/` not swept for ISA leaks like `core/` was (it is also "kernel") — the new test must cover it.
- `analysis/` fused-vs-non-fused STFT bodies not compared for large-block duplication.
- The ~80 peak-tracker/STFT tuning macros in `config.h:409-501` were spot-checked, not swept per-macro.
- Header-move basename-uniqueness + `.inc` instantiation chain must be build-verified by the implementer (read-only mandate).
