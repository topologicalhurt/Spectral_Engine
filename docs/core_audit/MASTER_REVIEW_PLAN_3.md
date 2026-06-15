# Major-Patchset Review — Instance 3

Third adversarial pass against `AI.md` (15-point checklist), `AI_CANON.md` (19 rules),
and K&R ("The C Programming Language", 2nd ed.). Method: 13 finder agents (per-subsystem
deep reads + cross-cutting duplication / wiring / constants / docs / K&R lenses) →
per-finding adversarial verification → this synthesized ledger. 52 raw findings; each
classified below as **FIX** (landed/landing through the build→ctest→perf-gate loop),
**REJECT** (false positive or already-handled, with reason), or **DEFER** (real but
needs a maintainer decision or risks the m7 perf gate; recorded with a recommendation).

Independent grounding (run alongside the fan-out) corroborated the doc/comment-honesty
cluster and added `bench_vdsp_audit.c` unwired (IF-1).

Gate per wave: all 6 targets build + `ctest` + `pytest tests/tools/test_perf_gate.py`
(arm32 codegen identity). Safest-first ordering: docs/comments/provenance/dead-code
before any codegen-possible change; anything that could move the arm32 hot path is
isolated and flagged.

---

## Execution status — LANDED

Waves A–G (incl. F-1/F-2) executed, one green commit each, gated by all-6-targets build +
ctest + the m7-baseline perf gate throughout. ctest 26 → **28** (added `resource_path_canonical`;
`q_ladder_parity`/`proc_mask_honesty` already present). Commit range on `minimal` (after the
review-2 close at `2838a92`):
- **A** doc truth — ghost flag, dissolved dir, drift budget 5e-6→2e-6, canon-doc narration.
- **B** comment honesty — Metal codegen (not hand-mirror), accum DTCM-inert, resource_fs C-is-SSOT,
  consume_file host-only, dead-branch + pass-number + change-history comments.
- **C** constant provenance — LUT_AMP_SCALE, SPQ "SPQ1" magic, MQ tolerance, CPU-clock fallbacks.
- **D** dead-code deletion — OSC_SET_MODE, 3 Q15 macros, 5 consts, OSC_LUT_MASK, BoxChars/BOX_ASCII,
  daisy clock/Q2.14/cycle-budget clusters, backend_get_caps, 2 proc fields (all zero-consumer).
- **E** SSOT reuse — synth_cpu fade ACTIVE, dead UINT32_MAX check, omp clock-check, BYTES_TO_MB,
  df scope, python DEFAULT args derivation.
- **F-1** tests — 9 `test_core_pass*` renamed (rule 18), check.h hoist into 2 stragglers.
- **F-2** security — new `resource_path_canonical` class test (fail-on-bug verified).
- **G** behavioral/honesty — analysis_full instrumentation no longer aborts a valid analysis;
  embedded_arm_float marked the reserved no-op gate it is (config + CMake + matrix rows).
- plus **rdc-perf-02** — deleted PerfMetrics' 4 write-only fields + the peak_resident_mb naming lie.

REJECT confirmed by hand: **arch-cmsis-osc-02** (the `test_dormant_cmsis_oscillator_still_compiles`
pin compiles it `-DARM_MATH_CM7=1 -Werror`, so the `_Static_assert` fires — already-handled).

Then three more landed from the DEFER set after the status block:
- **core-infra-01** — `SPECTRAL_CACHE_ALIGN` now derives from the `SPECTRAL_CACHE_LINE` SSOT (mem.h
  is `#include`d above the block); host-only, fixes the simulate 32-vs-64 disagreement.
- **analysis-track-01** — deleted the orphaned chunked-streaming tracker API (`update_threshold` +
  the always-NULL `overlap_magsq_row` param + its unreachable branch); core_math tracker suite
  17/17 confirms byte-identical output.
- **IF-1 reclassified REJECT** — `bench_vdsp_audit.c` is *intentionally* unwired (links Accelerate,
  absent from the production build) with a documented manual recipe and compiles clean; not limbo.
  Investigating it surfaced a real stale ref: its (and the oscillator-backend plan's) pointer to
  `VDSP_MATH_ACCEL_AUDIT.md` was pre-archive — fixed to `archive/`.

Still DEFERRED (genuinely perf-gate-risk / firmware-only / structural — see below):
rdc-daisy-01 (firmware, background task filed), arch-arm32-04, xcut-knr-01, analysis-track-03,
arch-out-kernels-03/E2-D2, xcut-dup-03. Untouched low-value: xcut-dup-04 bench-clock DRY.
ctest 26 → **28**; perf gate green across all ~16 commits.

---

## Waves (FIX)

### Wave A — doc truth (comment-only; no code)
- **xcut-docs-01 [HIGH]** `AI.md:56,58` documents ghost flag `SPECTRAL_IS_EMULATOR` (0 hits in tree); real macro is `SPECTRAL_IS_EMBEDDED_SIM` (`spectral_guarantees.h:85`, `spectral_hash_resources_xx32_xx3.c:16`). The doc's emulator-guard recipe silently compiles to the wrong branch. → rename both occurrences. (MEMORY.md carries the same stale name — fix too.)
- **xcut-docs-02 [MED]** `AI_CANON.md:3` lists `spectral_engine/synth` — dissolved by the KERNEL_LAYOUT refactor. → update to `arch`/`drivers`.
- **xcut-docs-03 [MED]** `CORE_CONTRACTS.md:140` claims exact_trig budget `5e-6` but the gate `test_guarantees.c:47 BUDGET_SIN = 2.0e-6`. → doc must match the enforced gate (2e-6).
- **xcut-docs-04 [LOW]** `CORE_CONTRACTS.md:54,61,108,118,128,159` carry phase/pass narration (`Phase B`, `B0/B1/B2 landed`, `pass 248`) in a live canon doc (rule 18). → strip scaffolding, keep the durable facts.

### Wave B — code comment honesty (comment-only)
- **xcut-dup-01 / IF-2 [MED]** `spectral_osc_formulas.h:10-14,22-23` + `spectral_segment_math.h:7-9` tell maintainers to hand-mirror formula changes into "the Metal shader string in spectral_oscillator.c" (which has zero MSL) and "checks this version at compile time" (the _Static_assert was deleted). MSL is now codegen'd into `drivers/metal/spectral_osc_metal_generated.h` by `metal_osc.py`, verified by `verify_metal_osc`. → rewrite to the codegen+verify reality.
- **arch-arm32-01 [MED]** `spectral_synth_arm32.c:943` claims `accum[]` is "in DTCM for zero wait-state access", but `SPECTRAL_MEM_FAST` is inert in every compiled target (the file's own header lines 18-21 say so). → state intent without the false present-tense claim. (canon-6)
- **core-hash-resource-02 [MED]** `spectral_resource_fs.c:26-27,44-45` mandates manual sync with Python `compress_path()`, but Python now delegates to this C function via the ctypes bridge. → state the C side is the SSOT; Python inherits it.
- **core-hash-resource-04 [LOW]** `spectral_resource_fs.h:60-65` FNV-1a provenance points at `resource_hashes.py::file_id_from_path()`, which now calls the C bridge. → correct the provenance.
- **core-hash-resource-01 [MED]** `spectral_hash_xx32_xx3.h:84-87` / `.c:296-298` document a `consume_file()→BACKEND_UNAVAIL on embedded` contract the code never implements (STREAM descriptor is available on all targets). → doc-honest variant: the consume_file family is host-file-API only, no BACKEND_UNAVAIL return.
- **core-hash-resource-05 [LOW]** `spectral_hash_xx32_xx3.c:223-237` "use streaming if region > SIZE_MAX" branch is unreachable on every (64-bit) target that compiles it; comment frames it as active. → mark as the 32-bit-host-only guard it is.
- **tests-comment-04 [LOW]** `test_window_backend_parity.c:66-67` comment claims an endpoint assertion `compare()` does not perform. → delete (the max_dev sweep already covers i=N-1).
- **tests-rule18-03 [LOW]** `test_arm32_process.c:111,118` retain pass-number narration ("Pass 145 fixed…", "(pass 168)"). → rewrite to the behavioral fact.
- **xcut-consts-macros-07 [LOW]** `spectral_peak_track_internal.h:20` stale `/* Removed force-define */` edit-narration; line 28 `MADV_POPULATE_WRITE 23` lacks the Linux-UAPI provenance. → delete the narration, note the ABI value.
- **IF-3 [LOW]** `spectral_peak_track.c:783-785` narrates "the previous 16M-per-thread virtual allocation" (change-history in a comment, point 6). → state the current invariant.

### Wave C — constant provenance (comment-only)
- **core-infra-04 [LOW]** `spectral_consts.h:37-38` `SPECTRAL_LUT_AMP_SCALE 32700.0f` — magic value, rationale only lives in other files. → add the headroom provenance.
- **xcut-consts-macros-04 [LOW]** `spectral_segment_parser.h:57` `SPQ_FILE_MAGIC 0x31515053` — add the little-endian ASCII "SPQ1" note.
- **xcut-consts-macros-05 [LOW]** `spectral_proc_adaptive_track_density.c:49-51` fallback MQ tolerance `0.05f`/`1e-6f` lacks rationale. → add the derivation note.
- **xcut-consts-macros-06 [LOW]** `spectral_debug_embedded_arm.c:95-97` bare clock fallbacks `480000000u`/`168000000u`. → note chip/clock provenance.

### Wave D — dead-code / dead-define deletion (zero live consumers; verify grep before each)
- **core-qosc-01 [LOW]** `spectral_oscillator_dispatch.h:130-131` `OSC_SET_MODE` dead macro that multi-evaluates args. → delete.
- **core-qosc-02 [LOW]** `spectral_q.h:168-172,177,327` `Q15_HOT`, `SPECTRAL_Q15_INLINE_DEFINED`, `SPECTRAL_Q15_TYPES` dead defines. → delete.
- **core-qosc-03 [LOW]** `spectral_oscillator_dispatch.h:104-109` `OscDispatchMode` gap (0,1,3); value 2 decodes to silent scalar. → comment the reserved gap.
- **xcut-wiring-03 [LOW]** `spectral_consts.h:12,13,34-35,49` five dead math consts (`SPECTRAL_TWO_INV_PI`, `SPECTRAL_PI_SQ`, `SPECTRAL_Q30_SCALE`, `SPECTRAL_INV_Q30_SCALE`, `SPECTRAL_Q31_PER_RAD`). → delete.
- **xcut-wiring-04 [LOW]** `spectral_lut.h:20` `SPECTRAL_OSC_LUT_MASK` unused. → delete.
- **rdc-box-03 [MED]** `spectral_console.c:28-34` + `.h:63-77` `BoxChars`/`BOX_ASCII` dead infra; renderers hardcode glyphs. → delete the struct + const.
- **rdc-cyclebudget-05 [LOW]** `daisy_seed_config.h:130-136,91` dead in-C cycle-budget constants (perf model owns cycles). → delete.
- **rdc-q214-04 [MED]** `daisy_seed_config.h:78,81-83` dead Q2.14 stretch/ADC constants describing a fixed-point path the code does in float. → delete the cluster + its comment.
- **xcut-dup-02 [MED]** `daisy_seed_config.h:20` `DAISY_CPU_FREQ_HZ 480000000UL` is dead AND contradicts `daisy_seed_sdram.h:33 SPECTRAL_DAISY_CPU_HZ 400000000u` (the one the perf model parses). → delete the dead 480 copy.
- **core-infra-02 [MED]** `spectral_backend.c:139-150` + `.h:54-64` `spectral_backend_get_caps` + `SpectralBackendCaps` zero callers; `max_segments`/`max_output_len` never set/read. → delete (no public consumer).
- **xcut-consts-macros-03 [MED]** `spectral_processing_chain.c:44-45` `deterministic_residual_db=-60`, `psychoacoustic_margin_db=3` written into fields no stage reads. → delete the fields + writes.

### Wave E — SSOT reuse / value-identical dedup
- **core-synth-seg-01 / xcut-consts-macros-01 [MED]** `spectral_synth_cpu.c:17-21,387,401,413` reinvents the fade-profile selection as a private macro; `SPECTRAL_FADE_SAMPLES_ACTIVE` is the SSOT. → use ACTIVE (value-identical).
- **core-synth-seg-02 [LOW]** `spectral_synth_internal.h:145` always-false `num_segments > UINT32_MAX` (uint32_t). → drop the clause.
- **core-infra-01 [MED]** `spectral_config.h:686-701` `SPECTRAL_CACHE_ALIGN` (gated on EMBEDDED) disagrees with `spectral_mem.h SPECTRAL_CACHE_LINE` (gated on ARM_M7) in the simulate build (32 vs 64). → derive ALIGN from the SSOT CACHE_LINE.
- **core-infra-03 [MED]** `spectral_omp.h:24-28` `omp_get_wtime` drops `clock_gettime` return + assumes CLOCK_MONOTONIC, unlike its log-helper sibling. → guard + check.
- **rdc-vram-06 [LOW]** `spectral_cli_pipeline.c:1226` open-codes `/(1024.0*1024.0)`; `BYTES_TO_MB` is the SSOT. → use the macro.
- **xcut-dup-03 [LOW]** `spectral_peak_estimator.c:452-455` re-implements the canonical `[-pi,pi)` wrap. → hoist a neutral leaf `spectral_wrap_phase_pi` into `spectral_fast_math.h`; both layers call it.
- **xcut-knr-02 [LOW]** `convert_segments.c:323,336-353` `df` declared in outer scope but only used under `#if SPECTRAL_HAS_CHIRP` (unused-var in compact). → declare at use.
- **xcut-consts-macros-02 [MED]** `perf_profile.py:34` `DEFAULT_PERF_CLI_ARGS` duplicates `core/constants.py:18 DEFAULT_SUITE_BENCH_ARGS`. → derive from the single source.

### Wave F — tests & scaffolding
- **tests-rule18-01 [MED]** rename `tests/core_math/test_core_pass{11..20}_*.py` (9 files) dropping the pass-number narration (rule 18); pytest glob-discovers, no module imports them — `git mv`.
- **tests-dup-02 [MED]** `test_osc_recursive.c:22-23` + `test_arm32_process.c:26-27` carry the local `CHECK`/`g_fail` the Pass-261 hoist centralized into `tests/support/check.h`. → include the header.
- **xcut-dup-04 [LOW]** three `bench_*.c` hand-copy the CLOCK_MONOTONIC timespec-diff. → `tests/support/bench_clock.h`.
- **IF-1 [MED]** `tests/core_contracts/bench_vdsp_audit.c` is unwired (the two `bench_q15_*` have cmake targets). → wire a consistent EXCLUDE_FROM_ALL target (or delete).
- **core-hash-resource-03 [MED]** path-canonicalization security branches (`..` pop, dot-dot-space, trailing-dot/space strip, >255 RLE) have no class test; `verify_resource_hashes` is C-against-C. → add a golden test with literal expected bytes (point 5 class test).

### Wave G — behavioral (isolated, carefully gated)
- **analysis-full-02 [MED]** `spectral_analysis_full.c:21-27,49-54` a logging-only FFT-byte estimate sits inside the alloc/abort guard → an instrumentation overflow discards a successful allocation and returns empty. → move the estimate into the logging block.
- **xcut-wiring-02 [MED] + xcut-wiring-01 doc half [HIGH]** `embedded_arm_float` ≡ `embedded_arm` (no `#if SPECTRAL_EMBEDDED_FLOAT` consumer); `CMakeLists.txt:136-140` comment + the build-matrix rows (`utilities.cmake:226`, `Makefile:125`) falsely claim a float split. Honest-minimal (non-destructive): mark `SPECTRAL_EMBEDDED_FLOAT` a reserved/unimplemented gate at its definition and correct the false comment + matrix rows; do **not** delete the target. (Full wire-or-delete = DEFER.)

---

## REJECT (false positive / already-handled)
- **arch-cmsis-osc-02** — claims `spectral_oscillator_cmsis.c` is compiled by no target so its `_Static_assert` never runs. False: `test_dormant_cmsis_oscillator_still_compiles` (test_embedded_perf.py:604) cross-compiles it `-DARM_MATH_CM7=1 -Werror -fsyntax-only`; the assert fires. This is the review-2 E1 dormant-pin resolution. (No live caller is the *intended* dormant state.)
- *(guard against finder false positives the verifiers would have caught:* `SPECTRAL_DEFAULT_*`/`SEGMENT_SIZE`/`Q15_SEG_SIZE` "double-defines" are `#if/#else` per-profile branches, not duplicates; `analysis_fused.c` "Pass 1/pass1_max" is a two-pass-algorithm label, not dev-pass narration.)*

## DEFER (real, but maintainer decision or perf-gate risk — recommendation given)
- **rdc-daisy-01 [HIGH]** `daisy_seed_spectral.c:97-147` SD-card `.spq` load `f_read`s an UNTRUSTED file straight into the segment pool and hand-sets state, bypassing `spectral_arm32_validate_segment_data` (per-segment overflow, monotonic order, `MAX_ACTIVE` bound) AND the SDRAM data-sync barrier that `spectral_arm32_load` runs. `load_buffer` correctly delegates. **Recommend:** route SD load through `spectral_arm32_load` (or a shared validate+barrier helper) + add a crafted-SPQ rejection ctest mirroring `test_arm32_process.c:184/194`. **Why deferred:** the daisy `.c` is firmware-only — no host target compiles it (only the config *header* via `daisy-config-layout-test`), so a code fix can't be gate-verified on this machine; needs the embedded build / maintainer. Highest-severity finding — should be the first thing addressed when the firmware build is available.
- **arch-arm32-04 [LOW]** `spectral_synth_arm32.c:925` the M7 path hard-gates on `!ctx->osc_lut` → silent zero output, but the shipped M7 coupled oscillator is gather-free (LUT only feeds the generic `#else`). **Recommend:** gate the `osc_lut` requirement to `#if !SPECTRAL_ARM_M7`. **Why deferred:** shifts the hot-path early-return predicate → perf-gate risk; the "silent zero on no-LUT" is a defensive edge, not a live bug.
- **xcut-knr-01 [MED]** `spectral_peak_track.c:296-576` the non-fused candidate chain threads a 10-scalar per-frame clump through 5 helpers; `SpectralFrameContext` already carries it (the fused path uses it). **Recommend:** pass `ctx` like `run_fused_frame`. **Why deferred:** codegen-possible across the tracker hot path → perf-gate risk; large mechanical refactor better done as its own gated unit. (Prior review noted peak_track.c's K&R pass was only sampled — MASTER_REVIEW_PLAN_2.md:284.)
- **analysis-track-03 [LOW]** the SIMD local-max pre-scan is written 4× across `process` + `run_fused_frame`. **Recommend:** one `static SPECTRAL_FORCEINLINE` scan helper. **Why deferred:** codegen-possible, finder itself set fix_safe=false; needs byte-identical before/after proof.
- **analysis-track-01 [MED]** the chunked-streaming tracker API (`update_threshold` + `overlap_magsq_row`) is orphaned (only caller passes NULL; `update_threshold` zero callers). **Recommend:** delete the dead streaming branch + trim the header contract, OR add a multi-chunk ctest. **Why deferred:** touches the non-fused tracker path; decide delete-vs-keep-as-public with the maintainer.
- **arch-out-kernels-03 [MED]** `arch/ref/spectral_out_kernels.c` is compiled by no target and its scalar bodies duplicate the host TU — this is review-2 **E2** (HIGH) + **D2**, listed but never resolved. **Recommend:** D2 hoist (`core/spectral_out_kernels_q15.h` shared `static inline`) so ref keeps only its CMSIS `#if`, then either wire a bare-metal target or delete the dead manifest entry. **Why deferred:** structural decision touching the embedded out-kernel path; kept deliberately for a future bare-metal build, so the maintainer should ratify delete-vs-wire.
- ~~**rdc-perf-02 [MED]**~~ **LANDED** — deleted the four write-only `PerfMetrics` fields
  (`virtual_mb`/`tracked_allocs`/`peak_resident_mb`/`cpu_utilization`) and the `peak_resident_mb`
  naming lie; `perf_print` already recomputed from deltas + printed `g_peak_alloc` directly.

---

## Honesty / second-pass (unknown-unknowns)
- The per-finding **adversarial verification did not complete** (session usage limit hit mid-run; the synthesis agent failed). The 11 findings verified before the cutoff are confirmed; the remaining 41 were re-verified **by hand** here (reading the cited lines), which is thorough but single-eyes — a second adversarial pass over the DEFER set is warranted, especially rdc-daisy-01's exploitability and the embedded_float wire-vs-delete call.
- **Least-scrutinized areas:** the Metal/CUDA generated payload vs the C SSOT beyond the version pin; the IFFT/vDSP numeric paths (no finder slice went deep on `spectral_synth_ifft.c` math); CUDA driver header (`drivers/cuda/spectral_cuda.h`) — no CUDA build on this host. The daisy firmware path is unverifiable without the embedded toolchain.
- **Lowest-confidence confirmed items:** core-infra-01 (cache-align — confirm the simulate-build 32-vs-64 actually matters for any allocation), xcut-consts-macros-03 (the proc fields may be a deliberate forward contract surface).
