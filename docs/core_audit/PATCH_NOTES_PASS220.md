# Patch notes — Pass 220: C2 `spectral_vector_ops` width audit — DECLINED on evidence (refactor Thread C2)

## Scope

Thread C2 asks whether `spectral_vector_ops.c` (the host SIMDe vector ops behind the
inherently-float FFT/analysis stages) leaves width on the table — i.e. whether
parameterizing it to a wider tier would pay. Per the measure-first / decline-on-data
discipline (same shape as PASS210, PASS218, A4), this pass **audits the actual code and
declines**. Documentation only — no source, build, or test change.

## Finding: the file is already hand-parameterized to its max tier

`core/port/host/spectral_vector_ops.c` is not relying on auto-vectorization to find width —
every **called** hot op carries an explicit `#ifdef __AVX2__` **256-bit `simde__m256`** tier
above the **128-bit `simde__m128`** tier, with a scalar remainder tail:

| Op | 256-bit (AVX2) | 128-bit | Notes |
|----|----------------|---------|-------|
| `spectral_vmul` / `vadd` / `vsq` / `vsmul` | ✓ | ✓ | element-wise float |
| `spectral_vmax` / `vmaxmgv` | ✓ | ✓ | reduce-to-scalar max / max-abs |
| `spectral_vatan2` | ✓ | ✓ | 8-/4-wide polynomial atan2 |
| `spectral_magsq_phase` | ✓ | ✓ | **hot analysis op** (magsq + atan2 phase) |
| `spectral_magsq_only` | ✓ | ✓ | **hot analysis op** (max-scan pass) |
| `spectral_deinterleave` | — | ✓ | **dead — no callers** |
| `spectral_magsq_split` | — | ✓ | **dead — no callers** |

The build is `-march=native -mavx2 -mno-avx512f` (`cmake/host-config.cmake:20-23`). So:

- On the **dev Mac (NEON = 128)** `__AVX2__` is undefined, the 128-bit `simde__m128` tier
  runs, and SIMDe maps it to native NEON — **already the machine's maximal width**.
- On **x86 AVX2** the 256-bit tier activates — **already the cap** the build allows
  (`-mno-avx512f` deliberately forbids 512).

There is no width tier left to add below AVX-512, and AVX-512 is **C3** — a separate
decision point that needs `-mno-avx512f` lifted, maintainer sign-off, and AVX-512 silicon to
validate (none of which this pass touches).

## The only two un-256'd ops are dead code

`spectral_deinterleave` and `spectral_magsq_split` are the only functions without a 256-bit
tier. Grep across `spectral_engine/` finds **zero callers** — both appear only in their
`core/spectral_vector_ops.h` declarations. The live analysis hot path
(`analysis/spectral_analysis_fft.c:363,365`) calls `spectral_magsq_only` /
`spectral_magsq_phase`, which **already have** 256-bit tiers and inline their own 8-wide
deinterleave. So widening `deinterleave`/`magsq_split` would be optimizing unreachable code —
declined per [[minimal-decline-on-data]]. (Whether to *delete* the two dead helpers is a
separate dead-code question, out of this audit's scope.)

## Verdict: DECLINE — nothing to parameterize

C2's own decline criterion ("decline if it already reaches max width") is met, more strongly
than the plan anticipated: the ops are not merely auto-vectorized, they are **explicitly
hand-written at both the 128-bit and 256-bit (AVX2) tiers**. The dev Mac already runs them at
its maximal NEON-128 width; x86 already runs them at the AVX2-256 cap. The next width up is
AVX-512 = C3 (gated). No code change pays here.

## What changed

- **`docs/core_audit/PATCH_NOTES_PASS220.md`** (this file) and the Thread C status in
  `QTYPE_REFACTOR_PLAN.md`. **No production source, build, or test touched** — nothing to
  re-baseline.

## Verification

Code-grounded, not assumed (per [[avoid-assumptions]]): tier coverage read from
`spectral_vector_ops.c` (256-bit `simde__m256` blocks at lines 25-30/45-50/65-70/85-99/
122-139/163-171/193-245/352-434/528-563; 128-bit + scalar tails throughout); flags from
`cmake/host-config.cmake:20-23`; caller counts from `grep -rn` across `spectral_engine/`
(deinterleave/magsq_split = header-only; magsq_only/magsq_phase = `spectral_analysis_fft.c`).

## Status

**C2 — DECLINED on evidence.** The float vector ops are already at max width on both targets;
the only un-widened ops are dead code; the only remaining tier is AVX-512 (C3, gated). This
exhausts Thread C's **locally-doable** work. The remaining Thread C items are hard-gated and
handed back:

- **C1** (16×Q15@256 kernel) — its deliverable is *throughput*, unmeasurable on this NEON-128
  Mac; the plan's standing rule is "we will not ship a 256/512 kernel we never ran." Gated on
  an x86 (AVX2) build/CI.
- **C3** (lift `-mno-avx512f`, add a 512 tier) — decision point, needs maintainer sign-off and
  AVX-512 hardware.

Float stays the desktop default; the opt-in `--q15` island is unchanged. See [[qtype-domain]],
[[minimal-decline-on-data]], [[avoid-assumptions]], [[faster-path-should-default]].
