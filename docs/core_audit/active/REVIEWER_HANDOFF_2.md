# Reviewer Handoff 2 — status reconciliation (as of pass 249; §4/§5 annotated pass 250)

**STATUS: SUPERSEDED 2026-06-21 — historical snapshot (as-of ~pass 250; the tree is now pass 283+).**
Every §4 concern is resolved or correctly gated (§4.1 build footguns FIXED; §4.2 width clamp DONE;
§4.3 PRECISE_PHASE documented EXPERIMENTAL; §4.4 window convention data-gathered + maintainer-gated;
§4.5 GPU parity CLOSED by the `gpu_backend_parity` ctest; §4.6 hardware-gated). S1 (P3–P6) CLOSED
(`M7_PERF_MODEL_PLAN` archived); S2 done. **The biggest stale claim is §1/F: "F-stream untouched, no
crossover measured, no IFFT implemented" is now FLATLY WRONG** — the IFFT path is implemented
(`spectral_synth_ifft.c`, `spectral_synth_hybrid.c` + the wavetable hybrid), the crossover is measured
(≈7 partials), and only the F3 golden remains. Genuinely-open + host-doable remainder: the public
`spectral_kernel.h` 1.0 freeze (now overlaps the pass-283 renderer reframe — coordinate before
freezing) and S4 reproducible desktop perf-counter benchmarks (low-value per the closed perf
frontier). STFT-unify is a recorded DECLINE, not open work. Live status of record:
`PLAN_CLOSURE_LEDGER.md`.

> This continues `REVIEWER_HANDOFF.md`, which remains the **standing Campaign-3
> mandate** (S1–S5 work-streams + the F synthesis fork). Nothing here supersedes
> it. This document exists because a long run of work (passes 237–249) has landed
> since that mandate was written, and the next reviewer needs (a) an accurate map
> of what is now done vs still open, (b) the current regression-test contract, and
> (c) the new and enduring concerns that the mandate could not have anticipated.
> Read the mandate first; then this.

---

## 0. One-screen status

- **Build/test baseline:** `ctest` is **19 tests, all green** (pass 250 added
  `segment_payload_bounds`); desktop, `embedded_arm`, `embedded_arm_float`, and
  the host oracle build clean. `pytest tests/core_math` (16) and `pytest
  tests/tools` (34) green. (If `tests/tools` ever reports ~10 fewer tests, check
  for a silently module-skipped `test_vendor.py` — a Python upgrade can drop
  PyYAML; this happened between passes 247 and 250.)
- **Two adversarial kernel sweeps** ran (passes 248, 249), each a multi-lens
  finder fleet with **3-skeptic majority-refute** verification. Combined: **8
  confirmed defects fixed**, **13 refuted** (the verification is load-bearing —
  most plausible findings are wrong). Every confirmed lifecycle-spanning defect
  became a compiled regression test, proven to fail-on-bug.
- **The perf-measurement stack (S1/S4) is built and consolidated** into
  `spectral_tools` (passes 237–245); the **vendored-dependency system (S3-adjacent)
  was reworked** (pass 246); a **security+correctness pass over `tools/`** landed
  (pass 247). See `M7_PERF_MODEL_PLAN.md`, ADR-0002, ADR-0003.

---

## 1. What landed since the mandate (passes 237–249), by work-stream

**S1 — ARM/embedded perf to the instruction & byte level.** The measurement
problem the mandate flagged ("there is no hardware") now has a built answer:
a layered model under `tools/spectral_tools/performance/embedded/` —
codegen census + `llvm-mca` CortexM7Model (modeled cycles) + a QEMU
`mps2-an500` TCG-plugin counts rig (measured instructions/bytes/working-sets).
Decision record + fidelity contract: `M7_PERF_MODEL_PLAN.md`. A real on-target
build break was found and fixed along the way (`__smulbb` is not an ACLE
intrinsic). **Open:** P3 (mca validation microbenches), P4 (AN4891-calibrated
memory layer), P5 (retire-or-rederive the old `spectral_perf_model.c`), P6
(apply the model: loop-nest, SMLALD coverage, ITCM). Still no board.

**S2 — adversarial bug-checker fleet.** Done **twice** (passes 248, 249). The
8 confirmed defects are in §3. The fleet's own false-positive rate is high
(13/21 findings refuted), which is the expected calibration — trust the
majority-refute, not the raw finder output.

**S3 — refactor / dedup / wiring / architecture.** Partial. The `tools/`
tree was hardened (ADR-0002/0003), `tools/core_audit` rehomed to
`tests/core_math/harnesses/`, the vDSP window path removed (one fewer
duplicated formula). **Not done:** the public `spectral_kernel.h` 1.0 API,
the stub "method" tokens (`hybrid_render`, `serra_smith`/`johnston` no-ops),
the dual full/fused STFT path unification.

**S4 — benchmark redesign.** The embedded half is built (S1 stack, one CLI
`benchmark_workflow`, a measurement matrix `performance/matrix.py`). The
desktop `tests_all` parallel-build race the mandate named is **fixed** (pass
248). **Not done:** reproducible desktop perf-counter benchmarks tied to the
model; the broader "every fast path has an accurate benchmark" bar.

**S5 — maturity audit.** Not started (gated on S1–S4 per the mandate).

**F — synthesis-method fork (oscillator-bank vs inverse-FFT).** **Untouched —
still the dominant open algorithm decision** (`OPTIMISATION_PLAN.md` A5/F2). No
crossover measured, no IFFT path implemented. This is the highest-value open
item after the S1 model is calibrated enough to price it.

---

## 2. Plan reconciliation — stale/changed claims to NOT trust at face value

- `REVIEWER_HANDOFF.md:27` said "14 tests" — now **18** (updated in place).
- `REVIEWER_HANDOFF.md` line ~33 ("bare toolchain lacks newlib; use
  `-ffreestanding`") is **superseded**: a real newlib toolchain is now installed
  on demand via `python -m spectral_tools.testing.benchmark_workflow m7-bootstrap`
  (sha-pinned xPack). The freestanding-stub approach is gone.
- `REVIEWER_HANDOFF.md:183` calls the desktop `tests_all` parallel-build race "a
  known flake to fix" — **fixed** (pass 248; oscillator-compiling test targets are
  gated on `generate_metal_osc`).
- `CORE_CONTRACTS.md` window section — **reconciled** (pass 249). The COLA test's
  `SPECTRAL_USE_VDSP=0` is now redundant: the vDSP window path was removed, so all
  backends generate the symmetric form. Cross-backend parity is enforced by
  `window_backend_parity`.
- `CAMPAIGN_2_MASTER_PLAN.md` / `OPTIMISATION_PLAN.md` contain many "ctest 4/4" historical
  pass-notes — those are records, not current claims; the current count is 18.
- Vendored deps: any doc referencing `third_party/libs.txt` is stale — it is now
  `third_party/libs.yaml` managed by `python -m spectral_tools.vendor` (ADR-0003).

No active plan was found to *contradict* the landed code; the items above are
staleness, not conflicts. The mandate's S1–S5/F decomposition still holds.

---

## 3. The regression-test contract (the "test criteria for major bugs")

The mandate's rule — "every confirmed lifecycle-spanning defect becomes a
compiled regression test" — was followed. The 4 ctest targets added this session
and what each *permanently* protects:

| ctest | protects | fails if |
|---|---|---|
| `osc_formulas_domain` | the SSOT waveform formulas' finiteness/range guards; built under UBSan | a reintroduced `(int)scaled` overflow (the scalar INT_MAX UB) |
| `window_backend_parity` | windows are the symmetric SSOT on every backend (built `VDSP=1`) | a reintroduced vDSP/periodic window |
| `peak_interp_parabolic` | the real C parabolic interpolator's fast **and** overflow-fallback branches | a wrong-branch / wrong-sign / wrong-formula regression |
| `segment_endian_roundtrip` | the Segment endian swap covers all 16 words incl. `_pad_w` cubic | a serialized field skipped by the swap |

Plus: `osc_parity` gained a quantized **INT_MAX boundary case** (pass 249);
`test_full_fused_parity` was hardened (real silence assertion + a real print
counter); `tests/tools/test_resource_hash_roundtrip.py` asserts the embedded
file_id table matches the C runtime.

Each was verified by reverting the fix and observing the test fail. **Keep this
discipline**: a confirmed kernel defect without a fail-on-bug test is unfinished.

---

## 4. New and enduring concerns (carry these forward)

These are not in the original mandate and will not surface from the code alone.

1. **Build-system fragility — both footguns now resolved (2026-06-18).**
   - *Incremental staleness (auto-deps gap):* **does NOT reproduce** on the current
     CMake build — header edits are tracked via compiler depfiles (`-MMD -MP`).
     Re-measured: touching `core/spectral_config.h` reschedules **82** dependent TUs.
     The narrower residual hazard is a typedef **size/layout** change made
     mid-experiment (e.g. flipping a `#if 0` fallback back and forth): that can leave
     mixed-ABI objects from a prior partial build, so after such a change force
     `--clean-first`. (This — not a depfile gap — is what bit the pass-248/249 reverts
     and the phase-at-peaks `SpectralHalf` experiment.)
   - *Clean deletes committed generated files:* **FIXED** (commit `0f647138`). The two
     committed-yet-`GENERATED` files (`core/spectral_hash_resources_xx32_xx3.c`,
     `drivers/metal/spectral_osc_metal_generated.h`) were `add_custom_command` OUTPUTs,
     so CMake added them to the `clean` list and `--target clean` deleted them. Each
     OUTPUT is now a build-tree **stamp**; the generator still rewrites the committed
     file in place (gated on the same DEPENDS), so clean removes only the stamp.
   - Remaining (maintainer-gated, S4): whether the generated-in-source-tree pattern
     should be replaced by build-tree-only artifacts + a CI regen check is a policy
     call, not a local fix.

2. **`width` is validated only for finiteness, never upper-bounded**
   (`spectral_contracts.h:42`). Both INT_MAX boundary defects (scalar pass-248,
   SIMD pass-249) stem from an unconstrained `width` reaching `rads*width`. The
   guards are now correct, but an **upper clamp on `width` at validation** would
   be defense-in-depth and remove a whole class of boundary risk. Recommended.
   **DONE (pass 250):** `|width| ≤ SPECTRAL_SEGMENT_WIDTH_MAX` (2^24) enforced in
   `spectral_segment_payload_valid`; ctest `segment_payload_bounds`, fail-on-bug
   verified.

3. **`SPECTRAL_PRECISE_PHASE` is dormant.** The cubic-MQ phase path
   (`spectral_segment_set_cubic`, the `_pad_w` annotation, the precise-phase
   synth branch) is gated behind this flag, which **defaults off and is set by no
   build target**. So the cubic path is unbuilt and untested end-to-end. The
   pass-249 endian fix hardened its serialization, but the feature itself is
   experimental-only. Decide: wire+test it, or document it as explicitly
   experimental and stop carrying latent risk in it.
   **DECIDED (maintainer, 2026-06-11): documented as EXPERIMENTAL** at the flag's
   definition (`spectral_config.h`); its real fate (wire+golden-sign-off vs
   delete) is decided when the F stream is engaged (F3 cubic-phase interpolation).

4. **Window convention is a deferred deliberate decision.** Pass 248 unified all
   backends on the *documented symmetric (N-1)* window (lowest blast radius). But
   for STFT analysis the *periodic / DFT-even (N denominator)* convention is
   arguably more correct (it satisfies COLA; it is what numpy/scipy default to).
   Adopting periodic as the SSOT would shift **every** backend's analysis output
   and needs golden re-sign-off (mandate D2). This is left open on purpose — it
   is the maintainer's call, with data.
   **DATA GATHERED (pass 250)** — harness `tests/core_math/harnesses/
   window_convention_sweep.c` (+ pytest `test_window_convention_sweep.py`)
   measured, symmetric vs periodic: (a) periodic Hann/Hamming satisfy COLA to
   float noise (~1e-7 ripple) at every dividing hop, symmetric misses by O(1/N)
   (1.5e-3 @ N=1024 hop N/2, 3.8e-4 @ N=4096); (b) Blackman fails COLA at hop
   N/2 in BOTH conventions (its k=2 term needs ≥4 phases) — OLA there is
   hop ≤ N/4 regardless; (c) the log-parabolic estimator is convention-
   INSENSITIVE (RMS bin-offset error differs <0.6%, periodic marginally worse);
   (d) adopting periodic shifts coherent gain by ~+1/N (+0.1% amplitudes @1024)
   and raw peak-triplet magsq by up to ~0.9%. Reading: periodic buys exact COLA
   only — which matters iff the F stream adopts IFFT/OLA synthesis; it does
   NOT improve current analysis. Cheapest correct path: keep symmetric now and
   fold the convention switch (if any) into the F-stream golden re-sign-off so
   one sign-off covers both. Final call remains the maintainer's.

5. **GPU backends remain under-tested.** The pass-249 GPU/SIMD lens refuted its
   GPU divergence findings, but there is **no CUDA/Metal hardware parity test in
   ctest** — only the SIMD-vs-scalar CPU parity and the Metal-MSL-vs-C codegen
   verify. CPU↔GPU numerical parity is asserted by neither. When GPU hardware/CI
   is available, this is a real coverage gap (S2/S4).

6. **On-target frontier is still the S1 wall.** The perf model is the best
   obtainable without a board; every "modeled" number is tagged as such. WCET,
   ITCM placement, real cache/DMA cycles, and perf-model calibration all wait on
   hardware (or QEMU+cycle-model, evaluated and deferred). Do not let a modeled
   number be cited as measured.

---

## 5. Suggested next steps (non-binding)

`F (synthesis-method crossover)` is the highest-leverage open item but rides the
S1 model — so the order is roughly: **S1 P3→P4 (calibrate the model enough to
price synthesis)** → **F (measure osc-bank vs IFFT crossover, decide)** →
**S3 (freeze `spectral_kernel.h` 1.0, collapse the stub method tokens, unify the
STFT paths)** → **S4 (desktop perf-counter benchmarks; the build-reliability
items in §4.1)** → **S5 maturity audit**. The §4 concerns (2, 3, 4) are small,
self-contained hardening items that can be done opportunistically at any point.

**Pass-250 progress + maintainer direction (2026-06-12):** §4.2 done, §4.3
decided (experimental), §4.4 data gathered (decision pending); **S1 P3 closed**
(M7_PERF_MODEL_PLAN status — CortexM7Model body-throughput validated ≤1% on a
9-case sourced microbench set; per-kernel cyc/iter now carry stated deltas).
Next is **P4** (AN4891-calibrated memory layer; the full TRM PDF is now
available locally for it). The maintainer also sharpened the S1/P6 goal: treat
the M7 as an embedded platform whose execution units (FPU, both ALUs, MAC,
dual-issue slots, cache/TCM) should be *meaningfully saturated*; **CMSIS /
CMSIS-DSP and the embedded-SIMD strategy are under-integrated with the codebase
and explicitly in scope**; the end state is an architecture (mirrored from the
best desktop design, ported to ARM) settled well enough that compiler-flag
tinkering and hand assembly become a meaningful, final-stage consideration —
see the expanded P6 entry in M7_PERF_MODEL_PLAN.md.
