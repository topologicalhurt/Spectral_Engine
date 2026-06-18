# Kernel-hardening cleanup plan

**Opened 2026-06-18, branch `minimal`.** This is the single active plan for the next
stretch. Lifecycle (maintainer's stated flow): **author → implement fully (with new
tests + new AI.md/canon principles per finding) → audit against AI.md principles →
simplify pass → mark closed** (then archive per the docs-tier rules).

Scope = the seven concerns (I–VII) raised on 2026-06-18, aimed at a *very hardened
kernel*. Every phase below ends in a fail-on-bug test and, where it establishes a rule,
a durable canon entry — per AI_CANON §18 (tests assert behavior; contracts and their
SSOT change in the same patchset).

Grounding: authored from a 9-investigator research workflow (`wf_2eee1987`, 1.08M
subagent tokens, 8/9 findings — the embedded-contract agent died on a session limit, so
§III is synthesized from the shared-substrate + M7 findings and is the one section whose
code-grounding gets re-verified at implementation time). All file:line refs and the
measured numbers below come from that pass.

### Decisions locked (2026-06-18, maintainer)
1. **Order = A→F** (the recommended sequence below); each step build + ctest + m7-gate
   green before the next.
2. **x86 (incl. AVX-512) is a REAL deployment target.** Consequence: build the
   **portable SIMD fallbacks + the per-op width policy NOW** (they compile to NEON via
   SIMDe here, so they're host-testable on arm64). What stays gated on x86 CI is only the
   x86-*specific* kernels (C1 16×Q15@256, C3 512-bit) and the AVX-512 *downclock
   measurement* — those cannot be authored-and-measured on this arm64 box. The desktop
   `apply_magsq_scales` win (IV) ships **with** its portable fallback, not vDSP-only; the
   VI profiles table keeps a **portable baseline** profile alongside `-march=native`.

---

## 0. Cross-concern structure (read first)

These concerns are **not** seven independent work items. Three couplings collapse the
work and dictate sequencing:

1. **VII (warnings) ⊂ VI (build profiles).** The *only* warnings on this machine —
   88 on `desktop`, 362 on `tests_all` — are 100% `-Wunused-command-line-argument` from
   `-mavx2` and `-mno-avx512f` being sprayed on arm64 (`host-config.cmake:20-23`, no arch
   guard). Arch-gating those flags is simultaneously VII's warning fix and VI's
   "arch-gate ISA flags" phase. Do it **once**, in VI.

2. **II + III + the perf-accounting substrate are ONE observability subsystem.** There is
   no `SpectralStageReport`/`SpectralRunRecord` today; timing, alloc, fallback and
   stage-ran signals are scattered across ≥5 incompatible vocabularies on desktop and a
   separate ad-hoc set on embedded. The fix is a **single stage taxonomy + record**
   (one `SpectralStage` enum, one report struct) that desktop renders and embedded
   *extends* with MCU metrics. Build the substrate once; II is its desktop render, III is
   its embedded extension.

3. **V (logging) shares decision points with II's masks.** The places that must *log* a
   fallback/decline are exactly the places that must *set a stage-ran/fallback mask bit*.
   Instrument each decision point once, emitting both the log event and the mask bit.

4. **IV (optimization) depends on II (honest measurement).** The current CLI under-reports
   wall by 13× (see §II), so no optimization can be trusted until the report is honest.
   IV comes after II/III — which is also the user's stated ordering ("subsequent to III").

### Recommended phase order (maintainer sets the final order)

```
A. VI-arch-gate + VII warnings   → clean build for everything else        [small, low risk]
B. VI profiles-table SSOT        → centralized/transparent/extensible flags [med, gated by m7]
C. I  generated-artifact contract → lut_data under verify, registry, canon  [med, low risk]
D. II+III observability subsystem → stage enum + record; honest totals;     [large, the heart]
   wall/busy/idle; alloc/realloc/fault; ran+fallback masks; embedded extn
E. V  major-path logging          → contract + backfill + presence lint     [med, interleaves w/ D]
F. IV optimization                → desktop magsq vectorize (measured -13%);  [med; algo frontier
   M7 de-spill + carry-state eval; record algo frontier CLOSED-with-rationale  mostly settled]
```

A unifying canon document — **`reference/OBSERVABILITY_CONTRACT.md`** — is produced in D
and referenced by II/III/V. The build philosophy goes in **`reference/BUILD_PROFILES.md`**
(VI). Both are new reference docs (no plans), per the tier rules.

---

## I. Generated-artifact contract (assembly-in-source)

**The literal ask is already satisfied — but nothing enforces it.** No committed assembly
exists anywhere outside `third_party/` (verified: `git ls-files | grep -iE '\.(s|S|asm)$'`
= 0). The project's only generated `.s` files are the M7-census intermediates
(`build/perf_model/arm32_m7.s`, `kernel_wrappers.s`, `loop_*.s`) — correctly **ephemeral
measurement artifacts**, not build inputs, never committed.

**Current state — 3 committed generated files, 2 conventions:**
- `core/spectral_hash_resources_xx32_xx3.c` (← `resource_hashes.py`) — stamp-OUTPUT +
  `verify_resource_hashes`, regenerate-and-byte-compare. ✅ canonical pattern.
- `drivers/metal/spectral_osc_metal_generated.h` (← `metal_osc.py`) — stamp-OUTPUT +
  `verify_metal_osc`. ✅ canonical pattern.
- `core/spectral_lut_data.h` (← `lut_generator.py`, `#include`d by `spectral_lut.c:15`,
  **compiled**) — **the outlier: ZERO cmake wiring, no generate target, no verify, no
  content drift guard.** Only a `_Static_assert` on table *size* (`lut_generator.py:61`),
  not on the 4097 sine values. It can silently ship wrong samples.

**Gaps:** (1) `spectral_lut_data.h` unguarded; (2) no single generated-artifact
**registry** — the three files are discoverable only by grepping `AUTO-GENERATED`
banners; (3) no `generated/`-folder or `asm/` convention; (4) the build-tree-stamp pattern
(just landed) + the generated-vs-measurement distinction are institutional knowledge in
two `.cmake` comments, not canon.

**Phases:**
- **I.1** Bring `spectral_lut_data.h` under the stamp+verify pattern: a `run_lut.cmake.in`
  runner, build-tree `generate_lut.stamp` OUTPUT, `DEPENDS` on the script + the config
  header that fixes `SPECTRAL_OSC_LUT_BITS`, and a `verify_lut` target wired onto every
  target that compiles `spectral_lut.c`. Add `--mode verify` to `lut_generator.py`
  (mirror `metal_osc.py:474-479` `read_utf8_lf` byte-compare). *[M / low — committed bytes
  stay identical; verify proves it]*
- **I.2** Add a single generated-artifact registry: one cmake-readable module
  (`cmake/generated-artifacts.cmake`) listing `{path, generator, stamp, verify-target,
  gating-targets}` per artifact; the three generator `.cmake` files consume their tuple
  from it. *[M / low]*
- **I.3** Establish the folder + naming convention and **reserve `asm/`** in canon (do not
  pre-create an empty folder — measure-don't-build-speculative). *[S / low]*
- **I.4** Codify in `reference/` (and a one-line AI_CANON rule): every committed generated
  artifact has a named generator + `AUTO-GENERATED` banner + registry row + verify-on-build;
  custom-command OUTPUT for an in-source committed file MUST be a build-tree stamp
  (clean-safety); generated **build inputs** are committed+verified, generated
  **measurement intermediates** stay ephemeral. *[S / low]*

**Canon to lock in:** the four rules in I.4. **Tests:** `verify_lut` fail-on-bug
(hand-edit one LUT value → build FATALs); a registry-completeness test (every
`AUTO-GENERATED`-bannered file appears in the registry + has a `verify_*` target); a
clean-safety regression test (configure → `--target clean` → assert all three committed
generated files still exist — pins the stamp-OUTPUT invariant the footgun fix established).

**Decisions for you:** (a) physically relocate the 3 files into `generated/` subdirs
(cleaner; churns `#include` paths) **or** keep in place + rely on banners/registry
[*rec: keep in place*]; (b) registry as cmake-readable SSOT vs human doc vs both
[*rec: cmake-readable SSOT, doc generated from it*]; (c) whether `spectral_lut_data.h`
stays committed at all vs demoted to build-tree-only (it's only consumed on the flash-LUT
path) [*rec: keep committed + guarded*].

---

## II. CLI performance reporting (the report you don't trust — correctly)

**Measured proof it's dishonest** (single run, `desktop`, `sin_440hz.wav` 2.00s, Metal):
the headline `Total` (`pipeline.c:1335`) is a **sum of per-stage kernel timers**, not wall
time. It reported **Total 4.8ms** while the true wall span (first-BEGIN→last-END) was
**64.4ms — a 13.4× under-report.** Synth showed 1.5ms kernel vs **58.7ms wall** (39×;
Metal first-dispatch `init()` of ~57ms is invisible). "Realtime 418×" is computed off the
fake total. "Peak tracked 0.7 MB" vs **RSS 23 MB** — a manually-summed ~5-site subset that
reads 3% of real heap.

**Gaps vs your spec (every one confirmed):**
- No per-stage **wall** in the headline (kernel-only `t_fft/t_track/t_synth/t_norm/t_write`).
- No **wall / processing / idle** split. Idle (`wall − busy`) is computed nowhere.
- **Allocations** = a hand-maintained byte sum missing the STFT matrix, FFT resources, GPU
  buffers, tracker arrays. **Reallocations** uncounted — `spectral_peak_interp.c:186` even
  logs an "unexpected" realloc no counter sees; `spectral_osc_bandlimited.c:345` reallocs
  uncounted.
- **No fallback/contingency flags as data:** cache-miss-rebuild, GPU-aux-reuse-disabled
  (`pipeline.c:1013`), tile-metadata-mismatch (`pipeline.c:996`), Q15-fell-back-to-float
  (`pipeline.c:1199`) are prose-only.
- **No stage-ran mask:** analysis path (full/fused/chunked), effective backend, cache
  hit/build, whether the **IFFT-hybrid** path was taken, Q15/SIMD/scalar dispatch, which
  processing fixtures ran — none surfaced as machine-readable bits.
- Two unlabeled "totals" with different meanings (normal-mode summed-kernel 4.8ms vs
  cache-mode wall 9.2ms). `getrusage` reads user/sys only (`perf.c:78-81`) — **`ru_majflt`**
  (major page faults, the closest proxy to "interrupts/costly paging") and ctx-switch
  counts are free in the same struct and ignored.

**Phases** (built on the §D substrate):
- **II.1** Define **`SpectralRunRecord`** — the single source of timing truth per run; CLI
  renders it, Python bench parses it, neither recomputes totals. *[M / low, additive]*
- **II.2** Make `Total` + `realtime_x` **wall** (monotonic first-BEGIN→last-END); report
  kernel/busy sub-timings additionally, labeled "busy", never as the run total. *[S / low —
  the headline honesty fix]*
- **II.3** Per-stage **wall/busy/idle** in the render; give GPU `init()` its own
  accounted span so 57ms lands in idle/init, not vanishing. *[M / med — touches backends]*
- **II.4** Real **alloc + realloc + fault** accounting: route the known heap sites through
  counted helpers (or, minimal variant, add the free `ru_majflt`/ctx-switch rusage fields +
  count the key sites); print alloc_bytes, alloc_count, realloc_count, major_faults — or
  don't print "Peak tracked" at all. *[M / low-med]*
- **II.5** **Stage-ran mask + fallback mask** as a structured one-line record. *[M / low —
  read-only instrumentation of existing decision points]*
- **II.6** Align the Python bench to the new record; keep warm-median-with-spread doctrine;
  share the stage/fallback enum C↔Python by **generation or binary-parse (C-truth rule)**,
  never hand-duplicated (kills the `STAGE_RE_V1/V2` parser drift). *[M / med]*

**Canon:** total/realtime are wall; every stage reports wall+busy+idle; "Peak tracked"
reflects real heap or isn't printed; ran/fallback are a machine-readable mask; one
`SpectralRunRecord` is the timing SSOT; debuggers are never timers; stage/fallback enums
are C-truth-shared. **Tests:** wall-Total ≥ Σbusy and ≈ stage-marker wall (fail-on-bug
vs the current 13× gap); realtime_x = audio_dur/wall; alloc_bytes within a factor of RSS
delta (not 30× below); ran-mask flips CPU↔Metal and cache build→hit; fallback-mask sets
`Q15_FELL_BACK` on an asin/pwm `--q15` request; parser-contract test extending
`test_perf_gate.py`.

**Decisions for you:** render channel (stderr markers + a "Run record:" line *[rec]* vs
JSON-on-stdout); alloc depth (full counted-helpers vs minimal rusage+key-sites *[rec:
minimal first, expand if needed]*); GPU init as its own stage vs synth-idle *[rec: own
stage — 57ms is too big to hide]*.

---

## III. Embedded / ARM observability contract (lock it early)

*(Section synthesized from the shared substrate + M7 findings; its code-grounding is the
one item to re-verify first in implementation — the dedicated investigator died mid-run.)*

**Why lock a contract now:** today the embedded picture is ad-hoc — `s_perf_stats` named
`FunctionProfile` slots (one, `amplitude`, is **dead** — declared `spectral_synth_arm32.c:237`,
never written), a separate `SpectralPerfCounters` (5 fields) populated **only** under
`SPECTRAL_RESTRICTED_PROFILE`, DWT/CYCCNT reads, and the frozen `m7_baseline.json`
(`process_insns`, cyc/voice-sample, code/data lines, WCET scenarios). There is no single
record an embedded engineer can read to answer "what is the firmware doing right now."

**What an embedded engineer + tester always needs (the contract to lock):**
- **Per-block WCET vs audio budget** — cycles consumed per `AUDIO_BLOCK` vs the budget at
  the configured sample rate; the headroom fraction. (The WCET stack already models this;
  surface it as a runtime/asserted field, not just an offline census.)
- **Cycle counts per kernel** (synth_core/fade/pair, the coupled-NCO step) via DWT.
- **Memory high-water:** DTCM / SDRAM / flash usage, **stack high-water**, and a
  **heap-must-be-zero** assertion (the engine is no-malloc on the RT path).
- **DMA / double-buffer state**, **underrun / IRQ counters**.
- **Active-voice count** and the no-realloc contract as a hard `== 0` assertion.

**Phases:**
- **III.1** Re-verify the current embedded instrumentation against the code (the one
  re-grounding step), then define the **embedded extension** of the §D `SpectralStageReport`:
  the same stage enum, plus MCU fields (DWT cycles/stage, DTCM/SDRAM/accum traffic,
  stack high-water, underrun count). Kill the dead `amplitude` slot. *[M / med — gated by
  the m7 perf gate: any field write on the arm32 path risks the label-pinned baseline; keep
  writes behind `SPECTRAL_RESTRICTED_PROFILE`/`SPECTRAL_NO_PERF` so default codegen is
  byte-identical]*
- **III.2** A **host-runnable QEMU harness** that asserts the contract (extend
  `test_perf_gate.py`): per-block budget fraction ≤ ceiling, heap-zero, no-realloc==0,
  cycles/kernel within bands. Distinguish QEMU-measurable (insns, cycles-modeled, memory
  sizes) from **real-hardware-gated** (true cycle timing, DMA/IRQ, board stack-paint).
- **III.3** Write **`reference/OBSERVABILITY_CONTRACT.md`** (shared with II) — the canonical
  desktop+embedded telemetry record, what's always reported, and the measurable-vs-gated
  split. Add the matching CORE_CONTRACTS row.

**Canon:** one stage taxonomy across desktop+embedded; embedded **adds** metrics to the
same report; no-realloc is a structural `== 0` perf-contract (supersedes the bare
`LOG_WARN`); instrumentation is zero-cost when off (gates preserve the M7-pinned codegen);
perf accounting is **measured-only** — never an in-C cycle projection (estimates come only
from the llvm-mca/QEMU stack). **Tests:** the QEMU contract harness (III.2);
`test_stage_report_embedded_parity` (restricted-profile unified report's OSC/SCAN
work-units == legacy `SpectralPerfCounters`); `test_stage_report_noperf_byte_identical`
(`SPECTRAL_NO_PERF=1` → rendered audio byte-identical, proving zero instrumentation effect).

**Decisions for you:** report always-on (behind `SPECTRAL_NO_PERF` only) vs opt-in (like
`SPECTRAL_TRACK_DEBUG_TIMING`, default off) *[rec: opt-in on embedded to protect the m7
baseline, always-on on desktop]*; `declined_reason` as a compact `uint8` code vs the
existing `SPECTRAL_RESOLUTION_REASON_*` strings *[rec: uint8 on embedded, string render on
desktop]*; whether adding any counted field to the arm32 path is worth a one-time
`m7_baseline` regeneration (label shift) or must stay strictly gated *[rec: strictly gated,
no baseline churn]*.

---

## IV. Hot-path findings — algorithmic first, then micro-opt

### IV-desktop — a real, measured win the "frontier" verdict missed
The 2026-06-18 `DESKTOP_PERF_CAMPAIGN` declared "no do-now speed win" but never attributed
the FFT stage's **~28% non-transform tail**. Measured (`shakespeare…wav`, n_fft=4096
hop=128, min-of-15 single-thread): **baseline 99.2ms → vectorize `apply_magsq_scales`
86.0ms (−13.3%)**; 8-thread FFT-stage −9.3%. **`full_fused_parity` PASSES with the change;
segment counts identical at −20/−40/−60 dBFS.** The culprit: `spectral_fft_apply_magsq_scales`
(`spectral_analysis_fft.c:93-130`) does scalar, double-precision, **branchy per-bin** work
on inputs that are provably finite≥0 — defeating auto-vectorization.

- **IV-d.1** Vectorize the interior scale+max (keep the endpoint clamp); the measured
  −13% single-thread / −9% 8-thread win. *[S / low — `full_fused_parity` is the gate]*
- **IV-d.2** Fuse magsq with `vDSP_zvmags` (secondary, same fn). *[S / low]*
- **IV-d.3** Demote/remove the per-growth realloc `WARN` in the **parallel emit hot path**
  (`spectral_peak_interp.c:172-196`) — it fires on every buffer growth inside the OMP region.
  *[S / low — log-only]*
- **IV-d.4 (gated)** The bigger lever (drop the fp16 store entirely → −27.6%) and the
  FFTW/x86 scalar magsq path are **x86-CI-gated** (no x86 silicon here).

### IV-M7 — algorithmic frontier settled; remaining wins are micro-opt
Measured: `synth_core_m7/.L492` = **21.25 cyc/voice-sample**, IPC 1.264, mca RThroughput
215 ≪ actual 340 cyc/16 → **latency-bound by the serial coupled-oscillator recurrence**
(4 q31 muls/sample), not resource-bound. **124 stack-spill ops/iter (7.7/sample)**; a
spill-free 1-sample mca floor is ~18 cyc (~15% headroom). The "LUT sin" is **dead on the
real M7** (only the generic `#else` fallback uses it). `spectral_coupled_renorm` exists +
is CTest-characterized (`test_osc_recursive`, SNR 83-128 dB) but has **zero production
callers** — production re-seeds each block with the **f64 Taylor sine**, which is ~10% of
cost at 256-sample blocks but **~1/3 at the 48-sample Daisy default**.

- **IV-m.1** De-spill the sustain loop (GCC unroll-factor sweep — it re-unrolls 4→16
  regardless of the pragma). *[M / low — recurrence math unchanged]*
- **IV-m.2** Evaluate carry-state + fixed-point renorm instead of the per-block f64 re-seed
  (the small-block Daisy win). *[M / med — **gives up the uint32-phase-as-SSOT exact-frequency
  guarantee**; a maintainer-call trade, gated by an SNR/drift CTest at block=48]*
- **IV-m.3** Record the **algorithmic half as CLOSED-with-rationale**: coupled-form chosen,
  2cos/Reinsch rejected on Nyquist amplitude; remaining wins are micro-opt only. *[S / low —
  documentation]*
- **IV-m.4 (gated)** CMSIS-DSP / `arm_rfft_q31` measurement — insns QEMU-measurable, cycle
  + ITCM/DTCM-placement wins board-gated.

**Tests:** producer-equivalence unit test (vectorized `apply_magsq_scales` == scalar ref
over a magsq battery incl. DC/Nyquist, zeros, large-finite); `frame_max` regression
(vectorized `vDSP_maxv` == scalar interior max — it feeds the dB-threshold gate); a
perf-gate assertion on `synth_core_m7` insns/voice-sample (so de-spill is captured + a
regression caught); if IV-m.2 proceeds, a multi-block byte/SNR parity CTest at block=48.

**Decisions for you:** ship `apply_magsq_scales` vDSP-only or also write the portable SIMD
fallback now *[rec: vDSP-only now, portable rides x86 CI]*; does the −13% FFT-stage win
re-open `DESKTOP_PERF_CAMPAIGN` or get recorded as a frontier correction *[rec: frontier
correction + ship it]*; carry-state renorm vs the exact-frequency guarantee (IV-m.2) — the
one genuinely significant algorithmic trade *[rec: prototype + measure at block=48 before
deciding]*.

---

## V. Logging for every major path

**Measured coverage:** library structured-log call sites = **0** (only the CLI, 10 sites,
+ arm sim, 2). **~180 `return SPECTRAL_ERR_*` across 22 files, ~0 logged at the return.**
Fully silent majors: `spectral_synth_hybrid.c` (≥10 decline points), `spectral_synth_cpu.c`,
`spectral_synth_ifft.c`, `spectral_resource_fs.c`, FFT-init (`spectral_analysis_fft.c`
`fail:`→`return 0`), tracker create (~8 silent NULL exits), `analyze_audio` bad-params,
`spectral_analysis_shape_init` (which constraint failed is never said). FFT backend
identity (vDSP vs FFTW) is never logged. `log_check` enforces **channel** (no raw printf)
but not **presence** — a new silent `SPECTRAL_ERR` ships uncaught.

**Phases:**
- **V.1** Write the major-path logging contract into `reference/` (canon). *[S / low]*
- **V.2** Backfill the high-value silent **decision/error origins** using the existing
  structured helpers (`spectral_log_error_codef`/`_warn_codef`, `spectral_format_resolution_context`)
  — no third logging idiom. Cover: analysis path decision, FFT backend identity + init
  failure, tracker create/finalize failures, backend fallback, IFFT-hybrid engage/decline +
  reason, resource-load failures, per-stage chain errors. *[M / low-med — pure additive]*
- **V.3** Extend the lint from **channel → presence**: every originating `SPECTRAL_ERR`/
  decline return in a dispatch/pipeline **allowlist** has an adjacent log. *[M / med]*
- **V.4** A behavioral fail-on-bug ctest pinning decision/error logging (neuter a log →
  test fails). *[M / low]*

**Canon:** every stage logs entry once (INFO); every capability/dispatch decision +
fallback logs (INFO designed-path, WARN degraded); **hot kernels carry NO logging**
(AI_CANON:115 — the caller logs the aggregate); fixed level discipline (ERROR=originated
abort, WARN=degraded-but-proceeding, INFO=stage/path, DEBUG/TRACE=loop-grain); embedded RT
paths use only strippable `SPECTRAL_DBG`-class macros (always-on log symbols unreachable
from the M7 synth path). **Tests:** `analysis_path_decision_logs`, `backend_fallback_logs`,
`ifft_hybrid_decision_logs`, `error_origin_logs` (FFT-init + `analyze_audio(NULL)`),
`log_presence_lint` (self-failing), embedded no-always-on-log-symbol nm/symbol check.

**Decisions for you:** presence-lint FATAL vs advisory after backfill *[rec: advisory
first, FATAL once green]*; INFO stage-entry always-on vs runtime verbosity threshold *[rec:
add a runtime level threshold — always-on INFO is too chatty for CI/batch]*; rule scope
(dispatch/pipeline allowlist only vs all mid-layer libs) *[rec: allowlist first]*.

---

## VI. Build profiles + flags (centralize, make transparent, extensible)

**Current state:** flags live in **≥9 places** — `host-config.cmake` (host + cuda + sim
sets), `daisy-config.cmake` (firmware), `options.cmake` (switches + cache vars), root
`CMakeLists.txt`, `toolchains/arm-none-eabi-gcc.cmake`, + scattered `target_compile_options`
in `cmake/targets/*.cmake`. **No profiles table. No stated philosophy.** `-mavx2
-mno-avx512f` applied unconditionally to **all** hosts incl. arm64 (the warning source).
`CMAKE_BUILD_TYPE` is **decorative** — `-O3` + fast-math come from
`SPECTRAL_COMMON_COMPILE_OPTIONS` regardless of Debug/Release. Fast-math is decided in
**3 independent gates with 2 toggles**. `embedded_arm` vs `embedded_arm_float` are
flag/define-identical except one reserved unbranched define. The dead-strip link idiom is
**copy-pasted across 13 test cmake files**. The help/build-matrix strings are hand-written
and can drift from the real flags.

**Phases:**
- **VI.1 (= VII fix)** Arch-gate the x86 ISA flags (emit `-mavx2`/`-mno-avx512f` only when
  `CMAKE_SYSTEM_PROCESSOR` is x86) → kills 100% of warnings. *[S / low]*
- **VI.2** Introduce **`cmake/profiles.cmake`** as the SSOT: a profile→{opt flags, quality
  flags, defines, rationale} table; each target selects a profile by name;
  `host-config`/`daisy-config` keep only SDK/library discovery + apply helpers. **Must
  preserve firmware flag byte-identity** (the m7 perf gate is the guard). *[M / med]*
- **VI.3** Unify fast-math to **one profile knob** (not 3 gates × 2 toggles). *[S / low-med]*
- **VI.4** Factor the test link/section idiom into one helper. *[M / low — pure refactor]*
- **VI.5** Make the build-matrix a **generated, flag-aware** printout derived from the
  profiles table (name→profile→flags→meaning), so it can't drift. *[M / low]*
- **VI.6** Document the two philosophies + the `CMAKE_BUILD_TYPE` caveat in
  `reference/BUILD_PROFILES.md`. *[S / low]*

**Canon:** build-flag SSOT = one profiles table; two philosophies as a contract (embedded =
aggressively minimal/fastest-for-target/deterministic-math; desktop = quality-but-optimized);
ISA-specific flags are arch-gated; one fast-math decision per profile; the build matrix is
generated from the SSOT; opt level is a profile property, not `CMAKE_BUILD_TYPE`. **Tests:**
configure-snapshot golden per build NAME (flag/define set); x86-flags-absent-on-arm64 /
present-on-x86; build-matrix derived-from-table (no drift); the m7 perf gate as the
byte-identity guard across the refactor; `embedded_arm` vs `embedded_arm_float` differ only
by the one define.

**Decisions for you:** `-march=native` (fastest-here) vs a pinned portable baseline
(distributable) *[rec: native for dev, leave a documented portable profile slot]*; give
`CMAKE_BUILD_TYPE` real meaning (a true `-O0/-Og` Debug profile) vs keep the doc caveat
*[rec: add a real Debug profile — actual debugging is currently impossible]*;
`embedded_arm_float` keep-as-reserved vs drop-until-needed *[rec: keep, documented]*; one
shared profiles module vs sibling host/firmware tables *[rec: one module, two sub-tables]*.

---

## VII. AVX-512, warnings, SIMD-width policy

**(b) Warnings — exactly one class, fully fixable.** Apple M1 Pro / arm64 / clang 15;
Release: **88 on `desktop`, 362 on `tests_all`**, 100% `-Wunused-command-line-argument`
from `-mavx2` (`host-config.cmake:22`) + `-mno-avx512f` (`:23`) sprayed on arm64. **No
source-level warnings.** Fix = **VI.1 arch-gate** → zero warnings. (Do it there; this is the
same fix.)

**(a) Why AVX-512 is off:** `-mno-avx512f` is a blunt global cap for **down-clock / hybrid-SIMD
safety** — on many cores 512-bit code lowers clocks and can net-lose. It is also
**unmeasurable on this NEON-128 box**, and **no 512-bit kernel exists** (lifting the flag
alone changes nothing until a kernel is authored — QTYPE Thread-C, x86-CI-gated). The reason
currently lives only in code/commit history, not canon.

**(c) SIMD width policy = widest-available UNLESS latency worse.** Today: the **float**
oscillator honors widest-available; the **Q15 pack8** kernel is **hard-pinned 8-wide@128**
and does not widen to 16×Q15@256 on AVX2 (C1, gated). There is **no latency-aware carve-out
in code** — width is chosen purely on ISA availability, and the one latency case (AVX-512
downclock) is encoded as the blunt global `-mno-avx512f`. `OSC_SIMD_WIDTH`
(`spectral_oscillator_dispatch.h:137-144`) is a **parallel hand-maintained width oracle**
separate from the `SIMDE_NATURAL_*_VECTOR_SIZE_GE` idiom the kernels use.

**Phases:**
- **VII.1 (= VI.1)** Arch-gate the x86 ISA flags → warnings clean. *[S / low]*
- **VII.2** Replace the blunt `-mno-avx512f` cap with a **capability + per-op latency
  policy**: width chosen per-op via `SIMDE_NATURAL_{FLOAT,INT}_VECTOR_SIZE_GE`, with an
  explicit downclock carve-out; retire the parallel `OSC_SIMD_WIDTH` oracle. The actual
  512-bit/256-Q15 **kernels stay x86-CI-gated** (won't author what we can't measure). *[M /
  med — policy now, kernels gated]*
- **VII.3** Document the AVX-512 decision + the width policy as canon. *[S / low]*

**Canon:** host ISA flags are capability-gated, never CPU-blind; SIMD width =
widest-available-unless-latency-worse, decided per-op via SIMDe natural-width predicates
(not a hand macro); AVX-512 (and any down-clocking width) is OFF by default, enabled per-op
only behind a measured net win — never globally/speculatively; build-flag rationale lives in
`reference/`, not commit messages. **Tests:** fresh Release build of `desktop` + `tests_all`
produces **zero** `-Wunused-command-line-argument` (fail-on-bug); cmake arch-gate assertion
(x86 flags absent on arm64, present on x86); extend `osc-width-parity` so the Q15 pack
kernel is bit-exact across 128/256 once a 256 tier exists.

**Decisions for you (all x86-CI / hardware gated):** C3 (lift `-mno-avx512f` + author a
512-bit tier) and C1 (16×Q15@256 on AVX2) — both need x86 silicon + a measured win, *do not
author speculatively*; on arm64, replace x86 `-march=native -mtune=native` with `-mcpu=native`
or drop *[rec: drop on arm64 — NEON already on]*; **is there a real x86 (AVX-512) deployment
target at all?** — this decides whether Threads C1/C3 are a priority or a latent option.

---

## VIII. Build reproducibility — ThinLTO + fast-math nondeterminism (CORRECTNESS, found during Phase B)

**Discovered 2026-06-18 while validating Phase B** (byte-identical compiler inputs, so NOT
caused by the profiles refactor). The default host build is `-flto=thin` + fast-math. On a
rare draw (~1 build in 13, observed under `-j8` `tests_all` load), the
`arm32_process_correctness` re-seed SINAD test produced **~28 dB vs a 70 dB floor** — a
*gross* miscompile of the host-compiled arm32 Q15 synthesis (the per-block f64 coupled-osc
re-seed path), not threshold jitter. HEAD passed 11/11, Phase B passed 12/13; the one failure
had compile_commands + flags.make + link.txt all byte-identical to a passing build. So it is
**ThinLTO's load-dependent, non-bit-reproducible parallel backend interacting with fast-math
reassociation** (and possibly a latent UB the optimizer then exploits) on that numeric path.

This is the single highest-priority finding to land (correctness-before-performance): the
*default* build can nondeterministically emit grossly-wrong audio. It overlaps §IV-M7 and the
`embedded-arch-audit` re-seed work.

**Phases:**
- **VIII.1** Root-cause: bisect whether it is (a) a real UB in the f64 re-seed under
  `-ffast-math` (run that path under UBSan/ASan + `-fno-fast-math` and a fixed ThinLTO cache
  `-Wl,-cache_path_lto`/`--thinlto-cache-dir`), or (b) pure ThinLTO codegen nondeterminism.
  *[M / investigation — the bisect is the work]*
- **VIII.2** Fix per the root cause: if UB, fix the re-seed math (likely the cheaper
  carry-state renorm from §IV-m.2 sidesteps the f64 path entirely); if ThinLTO-only, pin a
  reproducible-codegen flag for the affected TU or build the arm32 host-sim TUs without
  fast-math (they exist to mirror firmware, where SAFE_MATH is already the default). *[M / med]*
- **VIII.3** Make the test a **reproducibility gate**: run `arm32_process_correctness` over N
  fresh builds (or a fixed-seed ThinLTO cache) in CI so a nondeterministic draw fails loudly
  rather than ~1/13 of the time. *[S / low]*

**Canon:** the default build must be numerically deterministic — a flag combination that can
nondeterministically change *audible* output is a correctness defect, not a flake to retry.
**Tests:** an N-build reproducibility harness for the arm32 re-seed SINAD; UBSan coverage of
the f64 re-seed path.

---

## Appendix A — consolidated new tests (all fail-on-bug)

`verify_lut` · registry-completeness · clean-safety · wall-Total-honesty ·
realtime-from-wall · alloc-vs-RSS · stage-ran-mask · fallback-mask · perf-parser-contract ·
QEMU embedded-contract harness · stage-report-embedded-parity · stage-report-noperf-byte-identical ·
apply_magsq_scales producer-equivalence · frame_max regression · synth_core_m7 insns/voice-sample
gate · (gated) block=48 carry-state SNR/parity · analysis_path_decision_logs ·
backend_fallback_logs · ifft_hybrid_decision_logs · error_origin_logs · log_presence_lint ·
embedded no-always-on-log-symbol · configure-snapshot flag golden · x86-flags-arch-gate ·
build-matrix-derived-from-table · zero-warnings.

## Appendix B — consolidated new canon (reference/ + AI_CANON rules)

New reference docs: **`OBSERVABILITY_CONTRACT.md`** (II+III), **`BUILD_PROFILES.md`** (VI).
New AI_CANON one-liners: generated-artifact rules (I); wall-not-summed-kernel totals (II);
one stage taxonomy desktop+embedded + measured-only accounting + structural no-realloc (II/III);
major-path logging + level discipline (V); build-flag SSOT + arch-gated ISA + opt-is-profile (VI);
SIMD widest-unless-latency + AVX-512-off-by-default (VII).

## Appendix C — what stays GATED (not in this plan's local scope)

x86/AVX CI: the actual 256-Q15 (C1) and 512-bit (C3) kernels, FFTW/x86 magsq, the −27.6%
fp16-drop variant. Real hardware: CMSIS-DSP cycle/placement wins, DMA/IRQ/board telemetry,
true M7 cycle timing. These are recorded here as gated; the plan delivers the *policy,
contract, and host-measurable* parts and leaves the gated kernels for x86-CI / hardware.
