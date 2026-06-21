# Plan-closure ledger — full accounting of every plan item

**2026-06-18.** This is the master accounting for the directive *"close as many of the
plan items as possible; account for all the plans."* Every item in every plan under
`docs/core_audit/` (active, `reference/`, and `archive/`) was inventoried and given a
**definite disposition**: DONE, DECLINED, GATED, or CLOSED-this-pass. Nothing is left
"open and unaccounted-for".

Method: a 42-agent audit (per-plan inventory readers → adversarial verify → synthesis)
classified **238 items**. The adversarial pass demolished an optimistic first-cut
"closeable" list — 18 of the 33 raw-closeable items were actually already DONE (the
finders read pre-execution snapshots) or actually GATED. What truly survived as
*local + safe + decision-consistent* on this arm64 macOS box was small, and is now closed.

**Update 2026-06-21 — campaigns CLOSED since the 2026-06-18 audit** (recorded in the CHANGELOG, not
re-inventoried into the 238-item roll-up below, which stays the 06-18 baseline):
- **Kernel-hardening campaign** (CHANGELOG Pass 275) → CLOSED + archived
  (`archive/KERNEL_HARDENING_PLAN.md`): II CLI observability, VI/VII build-flag SSOT (`profiles.cmake`),
  §VIII ThinLTO determinism, V logging (AI_CANON #29), VII.3 width policy (#30). Gated remainder
  (IV-d magsq ULP, III embedded, VII.2, V.3, II.1) deferred-with-rationale — now in the archived plan.
- **Renderer-abstraction campaign** (CHANGELOG Pass 283) → synthesis reframed as rendering
  (AI_CANON **#31** + a CORE_CONTRACTS row); additive/wavetable/subtractive renderers + the wavetable
  IFFT hybrid landed + parity-tested (ctest #20–#25). F3-gated remainder: the hybrid dispatch routing
  + the host-only `synth → render` rename. Joins the §4.A maintainer-golden gate.
- **Embedded-perf** (CHANGELOG 276–282, the concurrent agent) → high-word-multiply `coupled_step` etc.;
  **512 voices now fit the EXACT per-partial oscillator bank**, REFUTING the IFFT plan's founding "512
  needs the IFFT" premise (premise-revision banner added to `IFFT_SYNTHESIS_PLAN.md`).
- **`spectral_kernel.h` 1.0 API freeze** (commits 70b5bd6 + d61f559) → DONE: the single public umbrella
  header (analyze_audio, synth dispatch, scene model, errors, I/O, wavetable, renderer registry) at
  `SPECTRAL_KERNEL_VERSION 1.0.0`, with the `kernel_api_freeze` ctest pinning the version + struct
  layouts + enum values + the curated entry-point signatures (a drift fails to compile). The
  freq-domain hybrids are explicitly PROVISIONAL (out of 1.0).
- **S5 maturity scorecard** (delivered inline at `REVIEWER_HANDOFF.md` §S5) → DONE: a 4-axis adversarial
  audit. Verdict — the algorithmic kernel is MATURE on every axis; productization is the gap. Grades:
  embedded device GAPS, 1.0 C library GAPS, reproducible reference kernel NEAR, desktop/GPU GAPS. The
  remaining-to-1.0 punch list is split host-doable (real library target + install, analyze_audio error
  channel, symbol namespacing, reproducible-build plumbing, GPU parity, throughput regression gate) /
  maintainer-gated (F3 golden) / hardware+CI-gated (on-target M7 bring-up, 128-voice WCET@48 + admission
  cap, real-GPU CI).
- The reviewer handoffs are RECONCILED / SUPERSEDED; **this ledger is the live status of record.** The
  two previously-live frontier items (the 1.0 freeze + S5) are now closed; the F3 golden + the S5 punch
  list are the remainder.

---

## 1. Roll-up — 238 items by disposition

| Plan group | DONE | DECLINED | GATED | CLOSEABLE |
|---|---:|---:|---:|---:|
| IFFT_SYNTHESIS_PLAN (F0–F7) | 11 | 1 | 6 | 2 |
| Campaign-3 mandate (REVIEWER_HANDOFF, S1–S5+F) | 8 | 0 | 8 | 1 |
| OPTIMISATION + DESKTOP_PERF campaign | 13 | 7 | 2 | 0 |
| M7_PERF_MODEL_PLAN (P0–P6) | 7 | 0 | 0 | 0 |
| OSCILLATOR_BACKEND_CONTRACT (Ph0–5) | 9 | 1 | 4 | 0 |
| QTYPE (refactor / domain / island) | 27 | 4 | 4 | 0 |
| phase-at-peaks (steps 1–15) | 14 | 0 | 1 | 0 |
| campaign2 + master-reviews 1/2/3 | 40 | 0 | 4 | 1 |
| audits + layout (ACS / ARCH / CLEANUP / LAYOUT / OSC-U / VDSP / GPU-Q15) | 52 | 0 | 11 | 0 |
| **Total** | **181** | **13** | **40** | **4** |

`181 done + 13 declined + 40 gated + 4 closeable = 238`. The "closeable" column is the
*surviving* count after adversarial verification (the raw first-cut was 33).

**Bottom line:** the tree is overwhelmingly DONE / DECLINED / GATED. The un-closeable
remainder is dominated by **one** maintainer gate — the **F3 golden listening sign-off**,
which transitively blocks the entire embedded-IFFT and GPU future — plus **x86/AVX CI**
and **real-hardware** (Cortex-M7 / Daisy / Apple-GPU / CUDA) walls.

---

## 2. Closed THIS pass

Two clean, local, decision-consistent wins + the doc-truth corrections. Commit
`0f647138` (build + envelope.h), plus this doc and the handoff/archive corrections.

- **Build footgun — FIXED** (REVIEWER_HANDOFF_2 §4.1 concern-1, deletion half).
  `core/spectral_hash_resources_xx32_xx3.c` and
  `drivers/metal/spectral_osc_metal_generated.h` are committed sources that were also
  `add_custom_command` OUTPUTs, so `cmake --build . --target clean` **deleted** them
  (reproduced). Each OUTPUT is now a build-tree **stamp**; the generator still rewrites
  the committed file in place, so clean removes only the stamp. Verified: committed
  files survive clean; generate+verify regenerate them byte-identical.
- **R2-D3 false-SSOT comment — FIXED** (`spectral_envelope.h`). The header claimed the
  three-region fade GEOMETRY is "factored HERE and nowhere else … Every backend reuses
  these." False — the arm32 integer backend re-derives the Q15 boundary math inline in
  both its inner loops. Comment corrected to the truth + rationale.
- **Doc-truth:** REVIEWER_HANDOFF_2 §4.1 rewritten — the *auto-deps staleness* footgun
  **does not reproduce** (header edits are depfile-tracked; a `config.h` touch reschedules
  82 TUs, re-measured); the residual hazard is the narrower typedef-size-change case.
  ARCHITECTURE_CLEANUP_STATUS.md's stale "Next recommended phase: G" marked LANDED.

---

## 3. Declined / deferred this pass (local, but not closed — with rationale)

- **R2-D3 geometry dedup — DECLINED ON DATA.** Hoisting the duplicated arm32 fade
  geometry into a force-inlined helper renumbers GCC `.L` labels and trips
  `test_perf_gate`'s live kernel extraction (`synth_core_m7/.L452` key lost) — measured.
  This is the *same* accepted-duplication reason already documented for the per-sample
  WALK: the M7 path's boundary math is part of the codegen the m7 perf gate pins.
  arm32.c left pristine; only the comment was corrected (above).
- **F2b dispatch wiring — DEFERRED (judgment).** Local + safe (a default-OFF compile
  gate keeps the default render byte-identical). Held deliberately at the **pre-F3
  stopping point**: the standalone, parity-tested `spectral_synth_hybrid_try_render`
  fast path is callable today (ctest `synth_hybrid_parity`); wiring it through
  `synth_cpu` adds a dispatch path whose only *end-to-end* validation is the gated F3
  golden audition, i.e. untested surface for zero default-path benefit. **Flip-ready**
  the moment F3 is scheduled. (An audit finder flagged this "closeable"; this is a
  conscious override of that optimism, consistent with the standing IFFT decision.)
- **F4a-(iii) static-twiddle ref FFT-backend pool API — DEFERRED (judgment).** Host-
  testable, but its sole consumer is the hardware-gated F4 embedded port; on host the
  ref backend works with malloc, so there is no host-realizable benefit to building the
  malloc-free pool infra now.
- **F2b mixed interior/edge seam-correction router — GATED (F3) + large.** ~200–300 LOC
  of correctness-critical boundary DSP with **no** production caller (rides F3). Building
  it now is speculative pre-F3 work.

---

## 4. Gated remainder (40 items) — organized by the gate that blocks them

### A. Maintainer decision / golden sign-off (a human must audition or pledge)
- **F3 golden listening test** — the keystone. Transitively blocks F2b default-on, F4c
  embedded default-on, F5 capacity republish, F7 GPU, and the whole offline IFFT tier
  (the IFFT path is an *approximation*, so default-on acceptance needs ears).
- **S3 public `spectral_kernel.h` 1.0 FREEZE** — the SemVer header already exists
  (`spectral_synth.h` / `daisy_seed_spectral.h` v0.0.1, deliberately WIP); a frozen 1.0
  ABI pledge is the maintainer's call.
- **S4.4 window convention** — data gathered (keep-symmetric favored); decision deferred
  into the F3 re-sign-off.
- **S5 maturity audit** — sequenced LAST after S1–S4; its API axis (kernel.h freeze) is
  itself unbuilt, so a defensible audit cannot be written yet.
- **OSC 2a-wiring** (would change shipped embedded output); **VDSP-PROMOTE `vvatan2f`**
  (byte-identity break); **GPU-Q15-GO `half2`** (−67 dBFS quality floor).
- **ACS-18/19** GPU-tile + tracker dedup (no dedup without a ratified parity harness);
  **ACS-20** ARM redesign (out-of-scope).
- **R2-D2** out-kernel dedup — the ref TU it would touch is compiled by **no** target,
  and review-3 defers delete-vs-hoist to the maintainer + a not-yet-existing bare-metal
  target.

### B. x86 / AVX CI (no x86 silicon on this arm64 box)
- **QTYPE** C1 (16×Q15@256), C3 (AVX-512 lift), Thread-C, lever-C.
- **OPTIMISATION** asm-tune-sine x86-unroll retry; **PERF-FFTW** `FFTW_MEASURE`.
- **phase-at-peaks step-15** FFTW/x86 runtime validation (the fp16 `_Float16` fallback
  path + FFTW deinterleave on non-Apple).

### C. Real hardware (Cortex-M7 / Daisy / Apple-GPU / CUDA — none present)
- **F4 embedded port as a whole** (F4a-i/ii + CMSIS half, F4b streaming OLA, F4c
  route + default-on), **F5** capacity republish, **F7** GPU FFT-inverse.
- **S1-P4** AN4891 memory-layer validation.
- **OSC 2b/3b-runtime** CMSIS + **GPU-Metal** runtime parity; **C2-A2/A3/A4** embedded
  redesign; **CUDA** parity.
- **S4** xctrace (needs full Xcode.app, not CLT) + Linux `perf` counters.

---

## 5. Where the active plans stand now

- **IFFT_SYNTHESIS_PLAN** — F0/F1/F2/F2b-step1 DONE; the standalone fast path is the
  clean pre-F3 stopping point. Everything past it (F2b wiring/mixed, F3, F4–F7) is gated
  on the F3 golden and/or real hardware. *No further local closure available without F3.*
- **REVIEWER_HANDOFF / _2** (Campaign-3 mandate) — S1/S2/S3 DONE; §4.1 footguns now both
  resolved (this pass). The remaining S-items are the maintainer/hardware-gated set above.
- All other plans are **archived** (`archive/`) or **reference** (`reference/`, no plans);
  their open tails are the gated items enumerated in §4.

**Conclusion:** every one of the 238 items now has a definite disposition. The locally-
closeable surface has been exhausted; what remains is gated on the F3 golden sign-off,
x86/AVX CI, or real hardware — none of which can be retired from this machine.
