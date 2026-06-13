# Major-Patchset Review — Instance 2 (Findings Ledger & Execution Plan)

## Execution status (branch `minimal`)

LANDED (each commit gated by all-targets build + ctest 24/24 + m7-baseline perf gate):
- W1 `comment scrub` — units lie (convert_segments Hz→rad/sample), stale-type lie
  (q63-as-Q30), canon-§6 prefetcher claim + first-person voice (peak_track), all
  surviving ULTRAPLAN/PASS/Thread-A/codename refs, filename-echo banners, atan2
  provenance.
- W2 `dead code` — NATIVE dispatch vocabulary + emptied oscillator_dispatch.c (deleted),
  CUDA alias macros, unreferenced getters, Pade consts + false GPU comment, 7 orphaned
  config defines, dead macro halves.
- W3 `Segment units` — per-field unit annotations on the core struct (AI_CANON §9).
- W4-D1 `eval-switch hoist` — the Q15 timbre→leaf switch (3 hand-copies) → one
  spectral_osc_q15_eval in spectral_oscillator_dispatch.h; codegen-neutral. CLOSES **E1**: the
  dormant CMSIS oscillator now has a fail-on-bug cross-compile pin (a real include-order
  -Werror break was found + fixed in the process).
- W5a `spectral_q15.{h,c} → spectral_q.{h,c}` — truthful Q-domain filename; pure
  path/guard rename, no symbol churn.
- W5b `oscillator.{c,h}/oscillator_dispatch.h → spectral_oscillator*` — the un-prefixed
  core group + non-namespaced guards; pure path/guard rename, codegen-identical.
- SD-5 `adaptive_track_density honesty` — was advertised "Implemented" but no-ops under
  the default build gate; now CLI lists it Experimental and the compiled-out path fails
  loudly (returns BACKEND_UNAVAIL) instead of silent OK. Cubic code kept for F-stream.
- W4-D7 `one ASCII tolower` — three lowercasers (one ASCII + two locale-dependent libc)
  → one spectral_ascii_tolower; latent locale-sensitivity in token matching removed.
- W5c `osc_* → spectral_osc_*` — the 18 exported oscillator API symbols (dispatch + SIMD
  interface + band-limited) renamed across 17 files; word-boundary, collision-checked;
  OSC_ macros / g_osc_ statics / ctest names deliberately left. SD-2 (files+symbols) DONE.
- A3 `spectral_osc_recursive.h → spectral_osc_q31.h` — the Q31 oscillator now named by
  domain, matching spectral_osc_q15.h; guard + includers updated (the test keeps its
  osc_recursive name — it pins the recursive-form numerical-stability contract).

REMAINING (priority order):
1. SD-3: rename the arch `oscillator_simd.{c}` pair (arch/simd = SIMDe, arch/arm = CMSIS;
   disambiguate the duplicate basename) + the arch/simd `oscillator_simd_*` companion files.
2. W4 D8 (DSB unify), D3/D5 (fade-loop factoring), D6 (sin_init parity).
3. W6 SD-4 `uq88_t`/`q30_t` typedefs; F1–F3 constant provenance; A4/A5 truthful symbol
   renames (`freq_inc`→`phase_inc`, `t_hop`→…); a proc-mask honesty ctest.
4. **Scoped follow-up** (maintainer-directed, its own campaign): grow spectral_q.h into
   the full cross-format conversion library (q15↔q31↔q63 hardware/bit-hack fast paths,
   arch-gated via macros), each conversion behind a parity test.

---

## Honest assessment of what the first pass left on the table

The first pass (W1–W7 + KERNEL_LAYOUT L0–L5) did real, durable work on the parts it actually
touched: lifecycle/allocation-failure discipline is genuinely strong (synth_cpu, synth_ifft,
analysis_full, and both tracker paths destroy on every error and check every alloc — no leaks
found), the GPU timbre cap was unified, the convert_segments /1000 unit bug was fixed, the MSL
mirrors were moved to codegen, and the core/arch/drivers move landed. But the maintainer's three
complaints are all substantiated, and the first pass left a large, coherent debt surface in
exactly the areas he named. The single biggest miss is **naming**: `spectral_q.h` is the home of
q15/q31/q63/uq16/uq32 (5 types, filename names 1) and was independently re-discovered by **five**
separate audit lenses; the entire `oscillator.*` family in `core/` is the only un-`spectral_`-prefixed
group in the kernel, with non-namespaced guards, re-discovered by **four** lenses. The comment
scrub the first pass claimed to complete is **not** complete — live `pass 145`, `pass 140`,
`PASS8`, `PASS200`, `ULTRAPLAN Phase A2/A3`, `Thread-A`, and a fleet of internal phase-codenames
(`Q3b/Q5b/Q5c/Bv/O1-B/B1`) survive in shipped code, plus a flat-out **units lie** ("freq > 255 Hz"
for a rad/sample threshold) in user-facing CLI output and a **stale-type lie** (a Q63 accumulator
documented as Q30, pointing the reader at the wrong helper). The K&R-on-hottest-kernels pass was
indeed deferred: the canonical Q15 eval switch is **hand-copied into three hot TUs** by the authors'
own admission, the three-region fade geometry is duplicated 4×, and `peak_track.c` is written in
first-person tutorial-blog voice and contains the *verbatim* canon-6-forbidden "prefetcher
completely hides latency" claim. Finally, wiring rot the first pass should have caught: an entire
uncompiled-by-any-target CMSIS oscillator file, a dead `NATIVE` dispatch vocabulary, dead CUDA alias
macros, a silently-no-op pipeline stage that the CLI advertises as "Implemented", and ~7 dead config
defines. This is not polish; it is the meat of a real review that the first pass skated past.

**Coverage honesty:** strongest in naming/comments/wiring/duplication around the oscillator+Q15
cluster (read line-by-line, multiply confirmed). Under-covered and flagged for a third pass:
analysis/ peak-body interiors (peak_track.c lines 60–780 and 1000–1500 only sampled), drivers/cuda
device-side math, seg_cache.c eviction lifecycle, and CLI/perf .c bodies — the comment-debt counts
there are lower bounds. No target was built and no ctest run during this audit; the dead-code
findings rest on static cmake source-list analysis + grep (conclusive for "no target lists the
file", but the CMSIS-Q15 verdict deserves a build to confirm).

---

## (A) NAMING & REFACTORING consistency

### A1 — `spectral_q.h` filename is a lie: it owns q15/q31/q63/uq16/uq32 [HIGH] — STRUCTURAL
- **file:line:** `spectral_engine/core/spectral_q.h:1` (banner "Q15 fixed-point types and arithmetic"); `:75-79` (`typedef int16_t q15_t; typedef int32_t q31_t; typedef int64_t q63_t; typedef uint16_t uq16_t; typedef uint32_t uq32_t;`); `:47-74` (Q-DOMAIN MAP listing 7 formats); owns `spectral_qadd32`/`spectral_smlald`/`spectral_q63_to_q15_scaled`. `spectral_q.c:1` carries the same lie.
- **rule:** AI.md item 13 + item 7; AI_CANON §9/§17
- **fix:** Rename to `spectral_fixed.h` (recommended) or `spectral_qtypes.h`; rename `spectral_q.c` in lockstep; move guard `SPECTRAL_Q15_H → SPECTRAL_FIXED_H`; rewrite banner to "Q-format fixed-point types and arithmetic (q15/q31/q63, uq16/uq32)". Keep all type names. The banner fix can land immediately at zero risk; the path rename is mechanical.
- **blast radius:** **19 files** `#include "spectral_q.h"` (grep-confirmed, across core/arch/analysis/cmd/tests + cmake source lists); public API `spectral_synth.h` does NOT include it (internal-only). Low risk, medium churn. *(The maintainer's flagship example. Merged from 5 lens reports: naming-files, naming-symbols, duplication, wiring-deadcode, architecture-lifecycle.)*

### A2 — `oscillator.*` family is the only un-`spectral_`-prefixed group in core/, with non-namespaced guards [HIGH] — STRUCTURAL
- **file:line:** `core/spectral_oscillator.c:1`, `core/spectral_oscillator.h:1-2` (`#ifndef OSCILLATOR_H`), `core/oscillator_dispatch.c:1`, `core/spectral_oscillator_dispatch.h:1` (`#ifndef OSCILLATOR_DISPATCH_H`), `arch/simd/oscillator_simd_scalar_waves.h:10` (`#ifndef OSCILLATOR_SIMD_SCALAR_WAVES_H`). Symbol leak: `osc_set_dispatch`, `osc_get_quality`, `osc_native_available`, `osc_simd_segment_*`, `osc_set_q15_enable` all bare `osc_` while the rest of the engine is uniformly `spectral_*`.
- **rule:** AI.md item 13 + item 7
- **fix:** Rename files to `spectral_oscillator.{c,h}` / `spectral_oscillator_dispatch.{c,h}`; guards to `SPECTRAL_OSCILLATOR_H` / `SPECTRAL_OSCILLATOR_DISPATCH_H` / `SPECTRAL_OSCILLATOR_SIMD_SCALAR_WAVES_H`; optionally rename `osc_*` public symbols → `spectral_osc_*`. The guard renames are zero-blast and can land first; the file+symbol rename is the structural decision (see SD-2).
- **blast radius:** **14 files** include `spectral_oscillator.h`, **10** include `spectral_oscillator_dispatch.h`; ~10 cmake targets + metal codegen + `osc_backend_contract.cmake` reference the paths; `osc_*` symbols span ~6 TUs + tests. Internal-only (public API unaffected). Medium-wide, mechanical; do as one commit. *(Merged from 4 lenses.)*

### A3 — `spectral_osc_q31.h` is named by technique but is the Q31 oscillator; sibling is domain-named [HIGH] — STRUCTURAL
- **file:line:** `core/spectral_osc_q31.h:1` ("Coupled-form (true-rotation) Q31 sinusoidal oscillator"); `:27-30` (`q31_t c; q31_t s;`, all primitives Q31). Sibling `spectral_osc_q15.h` is named by domain. No `spectral_osc_q31.h` exists.
- **rule:** AI.md item 13 + item 7 (the maintainer's domain-vs-technique example, named explicitly)
- **fix:** Rename to `spectral_osc_q31.h` (domain-consistent with `spectral_osc_q15.h`/`spectral_osc_formulas.h`); document the coupled-form technique inside; guard `SPECTRAL_OSC_RECURSIVE_H → SPECTRAL_OSC_Q31_H`.
- **blast radius:** **3 includers** (the lenses said 1–2; verified: `arch/arm/spectral_synth_arm32.c`, `tests/core_math/test_osc_recursive.c`, **`tools/spectral_tools/performance/embedded/native/qemu/qemu_main.c`** — the third was missed by the lenses) + `osc-recursive-test.cmake` target name. Low risk. *(Merged from 4 lenses; corrected blast radius.)*

### A4 — `freq_inc` names a phase increment (UQ0.32 per-sample phase step), not a frequency increment [HIGH] — STRUCTURAL
- **file:line:** `arch/arm/spectral_synth_arm32.c:976` (`uint32_t freq_inc = (uint32_t)seg->freq_q88 * ctx->freq_inc_scale_q24;`), `:981` (`phase_acc += sample_offset * freq_inc;` — added straight into the UQ0.32 phase accumulator). The file's own `:504` comment admits "freq_inc is a uint32 phase increment for a uint32 phase". Struct field `q31_t freq_inc;` (`spectral_q.h:309`); SoA array `q31_t freq_inc[]` (`arm32.h:45`). The genuine chirp slope is the separate `freq_delta` field — making `freq_inc` maximally confusing.
- **rule:** AI.md item 13; AI_CANON §9
- **fix:** Rename to `phase_inc` everywhere (local var, `SpectralActiveSegQ15.freq_inc` field, `active_soa.freq_inc[]`, `spectral_phase_batch4`/`spectral_coupled*` params).
- **blast radius:** ~20× in `spectral_synth_arm32.c` + struct field + SoA array; confined to the arm32 backend + one struct. Internal, low risk.

### A5 — `t_hop` is a sample offset but the `t_` prefix means wall-clock time everywhere else [MEDIUM] — STRUCTURAL
- **file:line:** `analysis/spectral_peak_track.h:61` (`SpectralFrameContext.t_hop`, a float sample position); `analysis/spectral_analysis_fused.c:14` (contract `t_hop = pair * hop`, hop in samples). The `t_` prefix elsewhere is timing: `t_synth`(105), `t_track`(60), `t_fft`(36), `t_analysis`(7), all CLOCK_MONOTONIC.
- **rule:** AI.md item 13; AI_CANON §9
- **fix:** Rename to `hop_offset` or `frame_start_sample`; reserve `t_` for timing.
- **blast radius:** ~49×, confined to analysis/. Internal, low risk.

### A6 — q88 (UQ8.8) and q30 (Q2.30) ride raw uint16_t/q31_t — self-admitted typedef debt [MEDIUM] — STRUCTURAL
- **file:line:** `core/spectral_q.h:62-74` (Q-domain map rows + admission "have NO dedicated typedef yet -- they ride uint16_t and q31_t, and the suffix is the only contract"); `:138-145`, `:263-265`; `SpectralSegmentQ15.freq_q88` (`:278`,`:293`). A `uq16_t` named `freq_q88` is type-indistinguishable from a phase-acc `uq16_t`.
- **rule:** AI.md item 13; AI_CANON §14
- **fix:** Add `typedef uint16_t uq88_t;` and `typedef q31_t q30_t;` (width-identical, no ABI change); use in `spectral_omega_to_q88` return, `SpectralSegmentQ15.freq_q88`, `spectral_q30_to_q15_scaled`. Replace the planning-narrative tail (see B-section) with a units statement.
- **blast radius:** No ABI change (same carriers). Touch `spectral_q.h/.c`, `SpectralSegmentQ15`, `SpectralActiveSegQ15`, `spectral_synth_arm32.c`, `perf_embedded`. Medium scope, low risk. *(Note: the first pass already tracks this in QTYPE_REFACTOR_PLAN; the maintainer named it as STILL-OPEN debt, so it is a live finding to pay, not just track.)*

### A7 — Duplicate basenames across arch/ subdirs: `oscillator_simd.c` ×2, `spectral_gpu_tile.c` ×2, `spectral_out_kernels.c` ×2 [MEDIUM] — STRUCTURAL
- **file:line:** `arch/arm/oscillator_simd.c:1` (CMSIS) + `arch/simd/oscillator_simd.c:1` (SIMDe); `arch/ref/spectral_gpu_tile.c` + `arch/simd/spectral_gpu_tile.c`; `arch/ref/spectral_out_kernels.c` + `arch/simd/spectral_out_kernels.c`. Listed in `source-manifest.cmake:57/59` etc. Distinguished only by directory; object files differ only by path; defeats grep/jump-to-file and stack-frame disambiguation.
- **rule:** AI.md item 13 / item 7 (the maintainer named "duplicate basenames across dirs")
- **fix:** Profile-suffix the basenames: `oscillator_simd_cmsis.c` / `oscillator_simd_simde.c`, `spectral_gpu_tile_ref.c` / `_simde.c`, `spectral_out_kernels_ref.c` / `_simde.c`. Update manifest lines + cross-reference banners. *(Note: the port-pattern, exactly-one-compiled, is a sanctioned framework — AI.md item 4 — so this is a disambiguation/decision, not a correctness defect. Some lenses correctly declined to file it as a hard violation. Surfaced here for the maintainer to ratify per SD-3.)*
- **blast radius:** Manifest entries only (these are `.c` TUs, not `#include`d by name) — a handful of cmake lines + 2 banner sentences per pair. Verify port-selection still resolves after rename.

### A8 — Segment fields `df`/`da` carry units (rad/sample², amp/sample) but have no unit comment at the struct definition [MEDIUM]
- **file:line:** `core/spectral_common.h:22` (`float start, length, phase, omega, df, amp, da;`), repeated `:57` and `:71` for SegmentCompact/SegmentGpu. Units live only implicitly in `SegmentLoopParams` (`synth_internal.h:23`), not at the canonical struct.
- **rule:** AI_CANON §9; AI.md item 13
- **fix:** **Comment-only** addition per field at the Segment definition (omega = rad/sample, df = rad/sample², da = amp/sample, start/length = samples), matching SegmentLoopParams style. A field *rename* would be high-blast (referenced across every backend), so the comment is the correct fix.
- **blast radius:** N/A (comment-only).

---

## (B) COMMENT doctrine

### B1 — `convert_segments` reports an omega (rad/sample) threshold as "Hz" — a units LIE in user-facing output [HIGH]
- **file:line:** `cmd/convert_segments.c:344` (`if (omega > 255.0f) stats.high_freq++;` — omega is rad/sample, fed to `OMEGA_TO_Q88`); `:370` (`"  %u segments freq > 255 Hz (encoded /4)"`). The `/4` part is correct; the "Hz" is the lie.
- **rule:** AI.md hard rule "keep units explicit"; AI_CANON §9; AI.md item 13
- **fix:** Change message to `"%u segments omega > 255 rad/sample (encoded /4)"`; rename stat field `high_freq → high_omega`. *(Note: `convert_segments_units` test pins the units — verify it doesn't assert on the string.)*

### B2 — Hot-loop output comment names the WRONG accumulator type and WRONG conversion function (Q30/`spectral_q30_to_q15_scaled` for q63 code) [HIGH]
- **file:line:** `arch/arm/spectral_synth_arm32.c:1210-1213` ("Convert the Q30 accumulator ... Q30 -> Q15 is a >>15 shift (see spectral_q30_to_q15_scaled)"). The accumulator is `static q63_t accum[256]` (`:921`); the actual call is `spectral_q63_to_q15_scaled(accum, ...)` (`:1213`). Mirrored at `spectral_q.h:263-264`, which also carries a banned `pass 145` reference.
- **rule:** AI.md item 6 + item 13; AI_CANON §6
- **fix:** Rewrite to "Pack the q63 accumulator (exact sum of Q15·Q15 products) to Q15 with the master gain: saturate >>15, then apply scale — `spectral_q63_to_q15_scaled`." Fix `spectral_q.h:263-264` for the same Q30/Q63 confusion + the pass reference.

### B3 — `peak_track.c` repeats the *verbatim* canon-6-forbidden "prefetcher completely hides latency" claim [HIGH]
- **file:line:** `analysis/spectral_peak_track.c:13-16` ("...perfectly triggers the CPU's L1/L2 hardware prefetcher, completely hiding DRAM load latency."); `:959-961` ("(which stalls the CPU pipeline)"). AI_CANON §6 lists "hardware prefetcher completely hides latency" *by name* as the canonical INCORRECT pattern. No test backs it; this is the hottest analysis kernel where the comment pass was explicitly deferred.
- **rule:** AI_CANON §6 + AI.md item 6
- **fix:** Delete the prefetcher/latency claim. If sequential access is load-bearing, state the design fact ("candidates are pushed in ascending bin order, so STFT reads are sequential") with no assertion about hardware, or tie it to a measured bench in `performance/matrix.py`.

### B4 — `peak_track.c` is written in first-person tutorial-blog voice [HIGH]
- **file:line:** `analysis/spectral_peak_track.c:1-21` ("We use AVX2/SSE", "we push", "we loop"); `:955-975` (decorated `=====` rule bars, first person, "We calculate these 3 boolean masks", "we found no peaks and instantly skip ... !"); `:1224` ("We copy the TrackSegment array..." — restates the next line); `:1437-1440`.
- **rule:** AI.md item 6; AI_CANON §6/§16
- **fix:** Rewrite to house style: neutral system-POV opener, then technical detail (the 3-condition local-max predicate, the movemask collapse) in third-person imperative. Delete WHAT-restating lines, the `====` bars, the exclamation.

### B5 — Residual pass/PASS/ULTRAPLAN change-history survives the "scrubbed" claim [HIGH]
- **file:line:** `core/spectral_q.c:12-16` ("former code ... halved every output sample ... caught by tests/arm_core, fixed in pass 145", "removed in pass 140"); `spectral_q.h:264` ("See spectral_q.c / pass 145."); `arch/simd/oscillator_simd_kernel.inc:24` ("the same <=1 ULP class already accepted at PASS200"); `analysis/spectral_analysis_fft.c:296` ("(PASS8: 2/Σwindow)"); `core/spectral_guarantees.h:1` ("(ULTRAPLAN Phase B1/B2)"); `runtime/spectral_perf_accounting.h:32` ("(ULTRAPLAN A2)"); `arch/arm/spectral_synth_arm32.c:14` ("NOT yet realized (ULTRAPLAN Phase A2/A3 ...)").
- **rule:** AI.md item 6; AI_CANON §18 (the prompt: any such reference found IS a live finding)
- **fix:** Strip every `pass N`/`PASS N`/`ULTRAPLAN Phase X` clause; keep the math WHY (q15.c keeps the >>15-vs->>16 −6 dB rationale; fft.c keeps "window amp scale 2/Σwindow, unscaled DFT"; kernel.inc keeps "<=1 ULP class pinned by the osc_width_parity ctest").

### B6 — Hot-kernel comments saturated with internal phase-codenames (Q3b/Q5b/Q5c/Bv/O1-B/B1) [HIGH]
- **file:line:** `core/spectral_oscillator.c:33,131,143,224,233` ("Q3b oracle, Q5b integer-NCO phase", "Packed 8×Q15 SIMD twin (Q5c)"); `spectral_oscillator_dispatch.h:135-137` ("plus sine (B1)"); `spectral_synth_cpu.c:31,260` ("Output-tiling (O1-B)"); `spectral_phase_nco8.h:1` ("(Q5c follow-up \"Bv\")"); `arch/simd/oscillator_simd.c:98-119`.
- **rule:** AI.md item 6; AI_CANON §18 ("state the fact, not the process that produced it")
- **fix:** Strip the bare codenames; keep the (otherwise good) system framing + math, at most one named plan-doc reference where a derivation lives. e.g. `spectral_oscillator.c:143` → "Scalar Q15-compute sustain path: integer-NCO cubic phase, Q15 waveform, float amp/fade".

### B7 — `daisy_seed_spectral.c` is wall-to-wall WHAT-comments + filename-echo dividers — doctrine never applied to the public Daisy API [HIGH]
- **file:line:** `api/daisy_seed/daisy_seed_spectral.c` — 28 bare WHAT-comments incl. `:29` "Initialize oscillator LUT once", `:41` "Set Daisy defaults", `:138` "Track memory usage", `:304` "Compute checksum" (over an XOR loop), `:333` "Checksum OK, execute command"; plus filename-echo dividers `:13` "Static memory pools", `:46` "Initialization", `:52` "SD Card", `:59` "Buffer Loading", `:62` "Parameters". Left untouched while sibling metal/cuda drivers were rewritten.
- **rule:** AI.md item 6; AI_CANON §6
- **fix:** Delete all WHAT-comments and section dividers; keep only genuine WHY (path-traversal rejection at `:403`, the non-const-pool note at `:129`). Functions are short and self-naming.

### B8 — `spectral_synth_cuda.cu` keeps legacy block-banner style + content-free dividers + unbacked claim [HIGH]
- **file:line:** `drivers/cuda/spectral_synth_cuda.cu:1` (loose prose banner, "No atomicAdd needed" — unbacked assertion), `:143` ("Public API" divider says nothing), `:32`/`:67` (struct WHAT-comments). Sibling `spectral_cuda.h`/`spectral_metal.h` were rewritten to dense system-POV; this driver was not.
- **rule:** AI.md item 6; AI_CANON §6/§16
- **fix:** Rewrite banner to system-POV-then-technical (driver role, tile→threadgroup contract, shared-memory cache invariant); delete the "Public API" divider and struct WHAT-comments; prove-or-drop "No atomicAdd needed".

### B9 — `SPECTRAL_PADE_SIN_C1/C2/C3` are dead and their comment "GPU shaders MUST use these same values" is false [HIGH]
- **file:line:** `core/spectral_consts.h:20-23` ("Pade [5/4] sine coefficients -- GPU shaders MUST use these same values"). Grep across the whole repo finds ZERO references to `PADE_SIN` (verified). `spectral_osc_formulas.h:63` states the Pade kernel was DROPPED. The constants are dead; the comment is a false single-source-of-truth claim.
- **rule:** AI_CANON §6; AI.md item 2
- **fix:** Delete `SPECTRAL_PADE_SIN_C1/C2/C3` and the comment. The canonical sine is the provenanced `SPECTRAL_MINIMAX_SIN_C3..C9`. *(Overlaps F-section dead-define cleanup; the false comment is the high-severity half.)*

### B10 — Live `TODO` planning narrative in hot kernels [MEDIUM]
- **file:line:** `arch/arm/spectral_synth_arm32.c:1277` (`/* TODO: DMA transfer whole struct or pack for UART/SWO */`); `core/spectral_hash_xx32_xx3.c:268-277` (`/* TODO: implement mmap-backed hashing ... When implemented: ... */` over a body that returns `SPECTRAL_ERR_BACKEND_UNAVAIL`).
- **rule:** AI.md item 6; AI_CANON §18
- **fix:** Delete the TODOs; track real work in the relevant `*_PLAN.md`. For the mmap stub: either drop the unimplemented method (it adds a dead enum slot + descriptor row no one can select) or replace the TODO with one units/why line; no future-tense recipe. (Fail-loud mechanism itself is correct per item 4.)

### B11 — `out_kernels.c` (both profiles) + `spectral_synth_simulation.c` carry the canonical "Find maximum.../Determine if..." WHAT-comment trio [MEDIUM]
- **file:line:** `arch/ref/spectral_out_kernels.c:54,73,77`; `arch/simd/spectral_out_kernels.c:43,56,60` (identical, duplicated with the body); `arch/arm/spectral_synth_simulation.c:123`.
- **rule:** AI.md item 6; AI_CANON §6
- **fix:** Delete the three WHAT-comments in both files + the simulation one. If any survives it must explain WHY (e.g. why `max_val > Q15_MAX/2` is the clip threshold).

### B12 — `spectral_arm32_process` + helpers studded with WHAT-comments and filename-echo dividers [MEDIUM]
- **file:line:** `arch/arm/spectral_synth_arm32.c:932,947,957,967,1020,1024,1044,1050,1196` (WHAT-comments: "Prefetch first few segments", "Track peak polyphony", "Compute block range", "Read current state"...); dividers `:482` "Initialization", `:546` "Loading", `:577` "Parameters", `:1229` "Interleaved Stereo Output", `:1252` "Restricted Mode Profiling"; loose banner `:42`.
- **rule:** AI.md item 6; AI_CANON §6
- **fix:** Delete the WHAT-comments and verb-dividers (names already say "process"/"load"/"init"); keep the dense WHY/derivation comments the file already has well.

### B13 — Extracted-TU banners narrate migration history ("extracted from", "former #else", "when it lived in") [MEDIUM]
- **file:line:** `arch/simd/spectral_gpu_tile.c:6` ("extracted from core/spectral_synth_internal.c ... its former #if ... body"); `arch/ref/spectral_gpu_tile.c:8` ("mirrors the former #else"); `arch/ref/spectral_out_kernels.c:4` + `arch/simd/spectral_out_kernels.c:4` ("extracted from core/spectral_out.c"); `drivers/metal/spectral_osc_metal_payload.c:6` ("when it lived in core/spectral_oscillator.c").
- **rule:** AI.md item 6; AI_CANON §18
- **fix:** Rewrite each banner to present-tense system role only; drop "extracted from"/"former"/"when it lived in".

### B14 — `spectral_osc_formulas.h` banner claims Metal constants are injected via `SPECTRAL_STR` in `spectral_oscillator.c` — false after L4 [MEDIUM]
- **file:line:** `core/spectral_osc_formulas.h:10-14`, `:21-23` (point at spectral_oscillator.c for Metal/SPECTRAL_STR/MSL strings). `spectral_oscillator.c` has ZERO such tokens (grep-verified); L4 moved MSL to `drivers/metal/spectral_osc_metal_payload.c` + codegen.
- **rule:** AI.md item 15 + item 6; AI_CANON §6
- **fix:** Rewrite banner: "Metal MSL is generated by metal-osc-codegen from this header; payload in `drivers/metal/spectral_osc_metal_generated.h`." Remove the spectral_oscillator.c claims.

### B15 — `spectral_oscillator.h`/`spectral_oscillator.c` banners are pure filename restatements [MEDIUM]
- **file:line:** `core/spectral_oscillator.h:1` ("Oscillator module"), `core/spectral_oscillator.c:1` ("Oscillator implementation") — the exact anti-pattern the maintainer named. Siblings `spectral_osc_formulas.h`/`_recursive.h`/`_q15.h` each state a real system role.
- **rule:** AI.md item 6
- **fix:** Replace with a substantive system banner (host-side oscillator dispatch hub: timbre→L0 X-macro map `SPECTRAL_OSC_TIMBRE_LIST`, the per-sample L1 `spectral_osc_eval` shared with CUDA, dispatch/Q15-enable setters; the hot per-segment loop lives in spectral_oscillator.c).

### B16 — atan2 polynomial coefficients ship ALWAYS-approximate with no provenance/error-bound [MEDIUM]
- **file:line:** `core/spectral_consts.h:25-29` (`SPECTRAL_ATAN2_A0/A1/A2` under bare "Polynomial atan2 coefficients"); `core/spectral_fast_math.h:31-44` (sharing story but no source/accuracy). Contrast the fully-provenanced `SPECTRAL_MINIMAX_SIN_C3..C9` ("~1.4 ULP vs libm") immediately adjacent — proving the inconsistency.
- **rule:** AI_CANON §16; AI.md item 3
- **fix:** Add a provenance comment: the fit (degree-2-in-s odd minimax of atan on [0,1]), measured max error in radians, a source. Mirror the SIN-coefficient depth. *(Lens noted: check `ACADEMIC_SOURCES.md` first; if sourced there, the fix is a one-line pointer.)*

### B17 — Low-severity comment polish (batch) [LOW]
- `core/spectral_config.h:595-598` — "formerly held 128 × 64-byte Segments" → present-tense budget.
- `core/spectral_q.h:70-74` — drop "Thread-A follow-up" planning tail (folds into A6 fix).
- `core/spectral_macros.h:11-13,25-27,54-56,69-71` — name-echo section dividers ("Loop Unrolling Hints", etc.); delete (keep the Utility-Macros multi-eval WHY at `:36-43`).
- `core/spectral_lut.c:37` — "now inline in spectral_lut.h" stale move-history; delete.
- `core/spectral_windows.c:1-4` — "X - X Implementation" filename-echo opener before the good system block; trim (apply to the other `X - X Implementation` openers for consistency).
- `runtime/spectral_perf.h:17`, `runtime/spectral_perf_accounting.h:5` — "The old X was retired because..." change-history; keep the forward invariant, delete the retirement narrative.
- `arch/arm/spectral_debug_embedded_arm.{h,c}` + `spectral_synth_arm32.h:7` — legacy `Features:`/`API Flow:` banners + content-free `/* Configuration */`,`/* Data Structures */`,`/* API Functions */` dividers; convert to system-POV matching the rewritten siblings.
- `runtime/spectral_console.h:13,41,53,60` + `runtime/spectral_utils.h:24,45,186,192` — boxed dividers that restate the group name; collapse/delete (keep one terse WHY where build-flag semantics matter).
- **rule:** AI.md item 6; AI_CANON §6/§18

---

## (C) K&R idiom

### C1 — `spectral_arm32_validate_segment_data` hides an O(n) sliding window with a re-derived end recompute [LOW]
- **file:line:** `arch/arm/spectral_synth_arm32.c:425-470` — the inner `while (first_live < i) { ... spectral_arm32_segment_end_checked_u32(...); if (first_end > start) break; first_live++; }` recomputes the first-live segment's end from scratch each outer iteration; four running scalars obscure the "how many prior segments still overlap" intent. Cold (load-time) path, freely improvable.
- **rule:** AI.md item 11 + item 9
- **fix:** Cache the running `first_live` end (it only advances); lift the overlap count into a named `active_overlap = i - first_live + 1` tied to `SPECTRAL_ARM32_MAX_ACTIVE` by comment.

*(Note: the deepest K&R debt is captured as DUPLICATION findings D1–D5 below — the triplicated eval switch, duplicated fade geometry, and repeated CMSIS prologue/tail ARE the K&R "structure over copy-paste, parameter clumps as structs" violations. The standalone idiom finding is light because the hottest kernels' idiom problems express as duplication. **Honesty:** a dedicated line-by-line K&R pass over `peak_track.c` (1602 lines, only sampled), `synth_arm32.c` (1309 lines), and the `.inc`/`.h` leaf math was NOT completed by any lens — the maintainer's complaint #3 remains under-covered and warrants a third pass.)*

---

## (D) Duplication

### D1 — The canonical Q15 timbre→evaluator switch is hand-copied into THREE hot TUs (self-admitted) [HIGH]
- **file:line:** `core/spectral_oscillator.c:132-141` (`osc_q15_eval`), `arch/simd/oscillator_simd.c:124-133` (`osc_q15_wave_scalar`), `arch/arm/oscillator_simd.c:260-269` (`osc_cmsis_q15_eval`) — byte-identical `switch(timbre){ case TIMBRE_SAW: return spectral_osc_q15_saw(pq); ... case TIMBRE_SINE: return spectral_osc_q15_sine(pq, lut); default: return Q15_ZERO; }`. The copies admit it: `oscillator_simd.c:122` "Mirrors spectral_oscillator.c's osc_q15_eval"; `arch/arm:259` "the embedded sibling of spectral_oscillator.c osc_q15_eval and host osc_q15_wave_scalar". The `SPECTRAL_OSC_Q15_VERSION` `_Static_assert` pins the *evaluators*, NOT this dispatch — three editor copies can drift undetected.
- **rule:** AI.md item 1; AI_CANON §17
- **fix:** Hoist one `static inline q15_t spectral_osc_q15_eval(q15_t pq, SpectralTimbre, const q15_t* lut)` into `spectral_osc_q15.h` (next to the evaluators); call from all three. Additive — all three TUs already include the header; delete the local copies. The `q15_simd_parity` ctest still pins numerics.
- **blast radius:** 3 callers, all already include the target header. `arch/arm` copy is `OSC_SIMD_CMSIS`-gated (not built on dev host) → `embedded_arm` must be in the verify loop. *(Merged from 3 lenses.)*

### D2 — `spectral_normalize_q15` + `spectral_mono_to_stereo_q15` scalar bodies are byte-identical across arch/ref and arch/simd [HIGH]
- **file:line:** `arch/simd/spectral_out_kernels.c:34-84` vs `arch/ref/spectral_out_kernels.c:45-123` — the only diff is the ref file wraps two spots in `#if SPECTRAL_USE_CMSIS` (arm_absmax_q15/arm_shift_q15) with the scalar in the `#else`; the host file IS that `#else` verbatim (absmax loop, the `while (test > Q15_MAX/2)` shift search, the `>>= shift_amt` loop all token-identical). The mono→stereo `SPECTRAL_UNROLL_4` loop is also duplicated.
- **rule:** AI.md item 1; AI_CANON §17
- **fix:** Extract the shared q15 scalar normalize + mono→stereo into `core/spectral_out_kernels_q15.h` (`static inline`) called by both profile TUs; ref keeps its CMSIS `#if` around the call.
- **blast radius:** 2 build-selected TUs; both already include the same headers. Behavior-preserving. Re-verify on `embedded_arm` (CMSIS branch). *(Note: also resolves the duplicate-basename concern's strongest case — see A7.)*

### D3 — Three-region fade geometry (`fi_end`/`fo_start`/`fade_step`/seeds) fully duplicated, M7 vs generic [MEDIUM]
- **file:line:** `arch/arm/spectral_synth_arm32.c:746-800` (`synth_segment_m7`) vs `:1116-1192` (generic `#else` inner loop) — identical `fade_step = Q15_MAX/fade_len`, fade-in clamp, `fo_start` clamp, fade-in seed `fade_val = seg_offset*fade_step`, fade-out seed `Q15_MAX - into_fade*fade_step`. The generic branch back-references the M7 version in a comment (`:1119`), proving the author knew it was a copy. A divergence is a silent fade-shape parity bug between M7 and the portable fallback.
- **rule:** AI.md item 1; AI_CANON §2
- **fix:** Extract `static inline SpectralFadeSpans fade_spans(seg_offset, blk_start, blk_end, fade_len, seg_length)` returning `{fi_end, fo_start, fade_step, fade_in_val, fade_out_val}`; both paths consume it and differ only in the per-region kernel. Intra-file hoist.

### D4 — ARM CMSIS oscillator repeats the same 5-local param-unpack prologue + j+=4 block + scalar tail in all 6 segment functions [MEDIUM]
- **file:line:** `arch/arm/oscillator_simd.c:43-201` (sine/saw/triangle/parabola/square) + `:279-315` (q15). `const float phase0 = lp->phase;` repeats 6× (`:48,83,117,138,174,285`) each with the identical `alpha/beta/amp0/d_amp` clump + identical scalar-tail loop; only the per-lane wave op differs. The 5-float clump is the textbook parameter clump that already lives in `SegmentLoopParams lp`.
- **rule:** AI.md item 1; AI.md item 11
- **fix:** Factor `osc_cmsis_fused(dst, lp, lane_wave_fn)` carrying the phase/amp pre-pass + accumulate, mirroring how `arch/simd/oscillator_simd.c` shares `oscillator_simd_scalar_waves.h` / the width-templated `.inc`. The 5–6 public symbols become thin timbre bindings (square stays apart — scalar sign).
- **blast radius:** Single CMSIS-only TU (not built on dev host → `embedded_arm` in verify loop).

### D5 — Three-region fade loop skeleton hand-rewritten in every CPU synth path [MEDIUM]
- **file:line:** `core/spectral_oscillator.c:94-127` (`synth_segment_scalar`) + `:159-187` (`synth_segment_q15`, labeled `:143-144` "Op-for-op the float synth_segment_scalar above"); `arch/simd/oscillator_simd.c:243-325`; `arch/simd/oscillator_simd_kernel.inc:336-405`. Same fade-in / sustain / fade-out traversal + the `(fade_out_start > fade_in_end ? ...)` clamp, fixed 4×; only the inner wave/amp expression differs.
- **rule:** AI.md item 1; AI.md item 7
- **fix:** Factor the region walk into a shared driver/macro that emits the three loops given a per-sample expression, so the boundary/clamp arithmetic lives once. Lower priority than D1 (inner bodies genuinely differ float/Q15/vector); the win is the boundary logic.

### D6 — `spectral_osc_q31.h` carries a third hand-written sine polynomial [MEDIUM]
- **file:line:** `core/spectral_osc_q31.h:58-67` (`spectral_osc_sin_init_f64` — degree-15 odd Taylor with literal factorials + private PI/INV_PI fold) duplicates the range-reduce-then-odd-polynomial structure of `spectral_osc_formulas.h:65-81` (`spectral_fast_sin_inline`) and the SIMD twin in `oscillator_simd_kernel.inc:154`. `peak_estimator.c:96` was specifically written to AVOID "divergent polynomial trig copies".
- **rule:** AI.md item 1; AI_CANON §17 + §7
- **fix:** Route the init through a shared double-precision sine helper, OR if f64 precision is required only here, add a parity test pinning `spectral_osc_sin_init_f64` against libm `sin` over `[-3π/2, 3π/2]` (`test_osc_recursive` currently tests rotation SNR, not the init polynomial).
- **blast radius:** Header included by 3 TUs (see A3) + cmake target. Low.

### D7 — Four ASCII-lowercase/CI implementations; two use the locale-dependent `tolower` the codebase bans [HIGH]
- **file:line:** `core/spectral_resource_fs.c:14` (canonical `spectral_to_lower`, comment "ctype tolower Deliberately not used", but file-static); `cmd/cli/spectral_cli_pipeline.c:51`; `analysis/spectral_processing_chain.c:55`; `runtime/spectral_utils.c:237` — the other three roll their own, two via locale-dependent `tolower()`.
- **rule:** AI_CANON §17
- **fix:** Promote `spectral_to_lower` to `spectral_utils.h`; convert the other three.
- **blast radius:** ~4 TUs, internal. *(Lens missed: also flagged 8 more kr-rest items it omitted for size — `path-p1/p2/p3` opaque naming at `resource_fs.c:49`, alias wrappers at `cli_pipeline.c:223` and `segment_parser.c:24`, duplicate validation ladders at `cli.c:80/107`, a 4th CI copy at `cli_pipeline.c:56`. These warrant the third-pass cmd/ sweep.)*

### D8 — Two different DSB-barrier emitters in one TU with divergent encodings [MEDIUM]
- **file:line:** `arch/arm/spectral_synth_arm32.c:58-60` (`#define __DSB() __asm volatile("dsb 0xF" ...)`, used by `spectral_arm32_dma_rx_sync:121`) vs `:395-401` (`spectral_data_sync_barrier` → bare `"dsb"`, used at init/load `:512/571`). Same ARMv7E-M full-system barrier spelled two ways; host fallbacks also diverge (`__sync_synchronize()` `:123` vs `__atomic_thread_fence(__ATOMIC_SEQ_CST)` `:399`).
- **rule:** AI.md item 1 + item 3 + item 11
- **fix:** Make `spectral_data_sync_barrier` the single primitive; have `spectral_arm32_dma_rx_sync` call it; pick one host fallback (the `__atomic_thread_fence`) for both.

---

## (E) Wiring / dead code

### E1 — `arch/arm/oscillator_simd.c` (entire CMSIS float+Q15 osc file) compiled by NO target [HIGH] — STRUCTURAL
- **file:line:** `arch/arm/oscillator_simd.c:1-318`. `source-manifest.cmake:58` defines `OSC_SIMD_EMBEDDED`; referenced ONLY at `utilities.cmake:76` (verified: inside `SPECTRAL_LOG_CHECK_FILES` — a *lint-only* file list, NOT an `add_executable`/`target_sources`) + `osc_backend_contract.cmake` (comment/glob). No Cortex-M cross-build exists (daisy uses `DAISY_ENGINE` = synth_arm32/debug_embedded_arm/q15/wavetable only). The file's own `:248-249` admits "it also has no LIVE caller yet". Its `_Static_assert` version pin (`:254-256`) is therefore never compiled, so `osc_backend_contract.cmake` presents dead code as contract-covered.
- **rule:** AI.md item 2 + item 4
- **fix:** Delete the file + its `OSC_SIMD_EMBEDDED` manifest entry (the contract glob then drops it cleanly), OR stand up the real Cortex-M cross-build / CMSIS host shim that compiles AND calls it. Do not leave a ~100-line uncompiled, uncalled kernel guarded only by a hardware `#if`.
- **blast radius:** Only a nonexistent CMSIS cross-build references it; host symbols come from `arch/simd/oscillator_simd.c`. Deletion safe for every green target. *(Merged from wiring + kr-hotkernels + architecture lenses; this is the maintainer's "left in limbo" item.)*

### E2 — `arch/ref/spectral_out_kernels.c` compiled by NO target [HIGH] — STRUCTURAL
- **file:line:** `arch/ref/spectral_out_kernels.c`. `source-manifest.cmake:80` defines `OUT_KERNELS_EMBEDDED`; referenced only at definition + `utilities.cmake:79` (verified: `SPECTRAL_LOG_CHECK_FILES` lint list). The HOST variant `arch/simd/spectral_out_kernels.c` is wired into every target. Manifest itself concedes not compiled by any current target.
- **rule:** AI.md item 2
- **fix:** Delete the file + its `EMBEDDED` manifest entry, OR add the bare-metal target that compiles it. *(If D2's hoist lands, the duplication collapses anyway.)*
- **blast radius:** No target links it; HOST file provides live symbols. Deletion safe.

### E3 — `adaptive_track_density` stage is a silent no-op in every shipped build, yet CLI says "Implemented" [HIGH]
- **file:line:** `analysis/spectral_proc_adaptive_track_density.c:85-129` (apply body wholly under `#if SPECTRAL_PRECISE_PHASE`; `else` at `:124-127` returns `SPECTRAL_OK` doing nothing). `SPECTRAL_PRECISE_PHASE` defaults 0 (`spectral_config.h:88-89`) and is set to 1 by **no** build/test (verified). `cmd/cli/spectral_cli.c:685` prints "Implemented adaptive_track_density"; `spectral_processing_chain.c:39-41` registers it in `k_stages` as real. A method that silently does nothing is worse than an error.
- **rule:** AI.md item 4 + item 2 + item 15
- **fix:** Move it to the Reserved set and stop `cli.c:685` claiming "Implemented", OR enable `SPECTRAL_PRECISE_PHASE=1` in a build/test so it is live and covered.

### E4 — NATIVE oscillator-backend dispatch fully vestigial [HIGH] — STRUCTURAL
- **file:line:** `core/spectral_oscillator_dispatch.h:78` (`OSC_MODE_NATIVE = 2`), `:107` (`OSC_DISPATCH_ALL_NATIVE`), `:159-160` (`osc_native_available`/`osc_set_native_available`), `oscillator_dispatch.c:9-11` (`g_native_backend_available`). Verified: ZERO set/read sites across spectral_engine+tests+api. Dispatch resolves only FALLBACK→SIMD/SCALAR (`spectral_oscillator.c:237-276`); no "native" backend exists.
- **rule:** AI.md item 2 + item 13; AI_CANON §19
- **fix:** Delete the enum value, the macro, the two accessors, the global. `oscillator_dispatch.c` is then a 0-line TU → also delete it + its manifest entry (see G2). A future device backend should be named for its device (GPU/CMSIS), not "native".
- **blast radius:** Zero external callers; pure removal. The 2-bit field still encodes 0–3 so dropping value 2 is source-only. Very low risk. *(Merged from naming + wiring lenses.)*

### E5 — Dead CUDA alias-only macros [HIGH]
- **file:line:** `core/spectral_oscillator.h:83-84` (`#define oscillator_normalize_phase_cuda spectral_normalize_phase`, `#define oscillator_fast_sin_cuda spectral_fast_sin_inline`). Grep over all c/h/cu/metal/py: zero uses (verified). Only `oscillator_cuda` (defined below) is called, from `spectral_synth_cuda.cu:132`. Pure renames with no callsite.
- **rule:** AI_CANON §19; AI.md item 2 + item 13
- **fix:** Delete both. The CUDA path calls `spectral_normalize_phase` / `spectral_fast_sin_inline` directly via `OSC_FORMULA_FUNC`.
- **blast radius:** N/A — no references, delete-only.

### E6 — Empty `#ifndef SPECTRAL_USE_EMBEDDED_SYNTH` / `#endif` block in a hot TU [MEDIUM]
- **file:line:** `core/spectral_synth_cpu.c:165-166` (verified literally empty: `#ifndef SPECTRAL_USE_EMBEDDED_SYNTH` immediately followed by `#endif`). Residue from a prior split.
- **rule:** AI.md item 2 + item 11
- **fix:** Delete the two lines.

### E7 — `osc_get_dispatch` / `osc_get_q15_enable` are write-only state (getters with no readers) [MEDIUM]
- **file:line:** `core/spectral_oscillator.c:31` + `spectral_oscillator.h:66`; `spectral_oscillator.c:70` + `spectral_oscillator.h:77`. Grep (excluding decl/def): empty across spectral_engine + tests. Setters are CLI-used; the getters are never read.
- **rule:** AI.md item 2
- **fix:** Delete both getters + declarations, OR add a set/get round-trip test reader.

### E8 — Every `#if SPECTRAL_PRECISE_PHASE` branch compiled by no configured build [MEDIUM]
- **file:line:** `core/spectral_config.h:88-89` (define 0, never set to 1 by any cmake/test — verified only a *comment* mention in `segment-endian-roundtrip-test.cmake`), gating `core/spectral_synth_internal.c:332` (cubic phase), `spectral_proc_adaptive_track_density.c:15`, `spectral_endian.h:53` (cubic-coeff endian swap).
- **rule:** AI.md item 2
- **fix:** Add a build/test defining `SPECTRAL_PRECISE_PHASE=1` so the cubic path is compiled/exercised, OR gate behind a tracked plan item and remove the dead apply-stage registration. *(Couples with E3.)*

### E9 — `osc_q15_wave_scalar` keeps a `TIMBRE_SINE` case the author says "never reaches this file" [LOW]
- **file:line:** `arch/simd/oscillator_simd.c:121-133` — comment "sine kept for safety even though sine never reaches this file" over `case TIMBRE_SINE: return spectral_osc_q15_sine(pq, lut);`. A "kept for safety" unreachable case is dead per item 2; the comment and code contradict.
- **rule:** AI.md item 2
- **fix:** Remove the case (default already returns Q15_ZERO), OR fix the comment if sine genuinely can dispatch here. *(Resolved automatically if D1's hoist lands — the shared eval keeps the case once, honestly.)*

---

## (F) Constants / macros

### F1 — arm32 kernel hardcodes block cap `256`, untethered to `SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE` [HIGH]
- **file:line:** `arch/arm/spectral_synth_arm32.c:894` (`num_samples=256u`), `:921` (`accum[256]`), `:1238` (`temp[256]`) — bare `256`; `spectral_config.h:514` names `SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE 256u`; no `_Static_assert` links them, so a config bump silently overflows `accum`/`temp`.
- **rule:** AI.md item 3; AI_CANON §19
- **fix:** Use the named cap for clamp + array sizes; add `_Static_assert(SPECTRAL_ARM32_MAX_BLOCK == SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE)`; replace the bare `256`s.

### F2 — Tracker/prefetch/page constants lack provenance while siblings carry `[chosen:]` notes [HIGH]
- **file:line:** `spectral_config.h:434` (`PREFETCH_LOOKAHEAD 12u`), `:449` (`ALLOC_FAILED_POLL_STRIDE 16u`), `:473` (`STFT_CHUNK_FRAMES 512u`), `:482` (`PRETOUCH_PAGE_SIZE 4096u`) — bare; siblings `CANDIDATE_BATCH [chosen:128]`, `GPU_TILE_SIZE`, `NORMALIZE_HEADROOM` carry provenance. `4096` also hardcodes page size (AI_CANON §8; Apple Silicon is 16 KiB).
- **rule:** AI.md item 3; AI_CANON §19 + §8
- **fix:** Add a `[chosen: ...]`/budget note to each, or mark not-swept; replace the `4096` page assumption with `sysconf(_SC_PAGESIZE)`.
- **note:** `STFT_CHUNK_FRAMES=512` here vs MEMORY.md's "4096" is a stale-doc divergence (AI.md item 15) — reconcile.

### F3 — Undocumented magic literals with no 2^k/format derivation [MEDIUM]
- **file:line:** `core/spectral_phase_nco8.h:52` (`th[25]` = 3·8+1, unstated); `core/spectral_q.h:133/144` (`65536.0f` = 2^16 phase, `256.0f` = 2^8 Q8.8, uncommented); `core/spectral_osc_q15.h:53` (`0.99996948` = 32767/32768, unexplained); `core/spectral_config.h:112` (`DMA_BATCH 32` ~512B, only correct for the 16B non-compact segment).
- **rule:** AI.md item 3; AI_CANON §19
- **fix:** Annotate each with its 2^k/format derivation or name it.

### F4 — Seven unconsumed defines in `spectral_config.h` (plus dead macro halves) [MEDIUM]
- **file:line:** `spectral_config.h:497` (`ERROR_BLINK_PAUSE_MS`), `:485` (`SEGMENT_POOL_BLOCK_SIZE`), `:419` (`TRACK_INTERP_LOG_DOMAIN`, sibling POWER_RATIONAL used), `:446` (`TRACK_PREFETCH_WRITE_LOCALITY`, READ used 5×), `:455` (`TRACK_SEG_PREFETCH_DISTANCE`), `:537` (`WAVETABLE_MASK`, wrap is via +1 guard sample not a mask), `:167` (`SAMPLE_HALF`). Also `Q15_SEG_SIZE` (`:151/154`). Plus `spectral_macros.h:29` (`SPECTRAL_LIKELY`, 0 refs vs UNLIKELY 8) and `:16` (`SPECTRAL_UNROLL_2`, 0 refs vs UNROLL_4 6).
- **rule:** AI.md item 2 + item 3
- **fix:** Delete each zero-consumer define + its per-compiler variants, OR wire into the site that warrants the hint/unroll. *(SPECTRAL_XSTR and PREFETCH_WRITE_LOCALITY are NOT dead — used internally; leave them.)*

### F5 — Low-severity macro polish [LOW]
- `core/spectral_oscillator.h:53` (`SPECTRAL_OSC_EVAL_CASE` hides `case ... return FN(rads,width)` and captures `rads`/`width` not args) — X-macro idiom, scoped/`#undef`'d at `:55`; note macro intent or pass `rads`/`width` via X args.
- **rule:** AI.md item 3; AI_CANON §19

---

## (G) Architecture / lifecycle

### G1 — Bare TODO + "When implemented:" planning narrative for an unimplemented mmap hashing method [MEDIUM]
- **file:line:** `core/spectral_hash_xx32_xx3.c:268-277`, `:289-295` — descriptor registers `FULL_MMAP` with `.available = 0`, body returns `SPECTRAL_ERR_BACKEND_UNAVAIL`. Fail-loud is correct (item 4); the TODO + future-tense recipe is the defect.
- **rule:** AI.md item 6 + item 4; AI_CANON §18
- **fix:** Drop the unimplemented method (it adds a dead enum slot + descriptor row no one can select) and track in a plan doc; OR, if the slot must stay for ABI, replace the TODO with one units/why line. *(Also listed under B10.)*

### G2 — `oscillator_dispatch.c` is a 12-line TU holding one global bool + two trivial accessors — over-granular split [MEDIUM] — STRUCTURAL
- **file:line:** `core/oscillator_dispatch.c:1-12` (entire TU: `g_native_backend_available` + the two native accessors). `source-manifest.cmake:29-30` lists both `spectral_oscillator.c` AND `oscillator_dispatch.c` in CORE for every profile — no port-pattern reason. The SIMD contract that justifies a separate file lives in the *header*. The maintainer named "the inverse: over-granular splits that should be merged."
- **rule:** AI.md item 7
- **fix:** If E4 lands (delete the native vocabulary), this TU becomes empty → delete it + its manifest line; the header `spectral_oscillator_dispatch.h` stays as the contract. If for some reason native is kept, fold the 3 lines into `spectral_oscillator.c` next to `g_osc_dispatch`.
- **blast radius:** 1 manifest entry + 3 lines; no symbol/API change; low risk. *(Couples with E4.)*

---

## Summary counts
- **HIGH: 25** (A1, A2, A3, A4, B1, B2, B3, B4, B5, B6, B7, B8, B9, D1, D2, D7, E1, E2, E3, E4, E5, F1, F2 — plus the two duplicate-confirmed lens entries already merged)
- **MEDIUM: 27**
- **LOW: 9**
