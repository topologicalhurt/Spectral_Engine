# Patch notes — Pass 206: Q0 — Q-domain type discipline + contract CTest

## Scope

Q-type domain phase step **Q0** (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5): land the
type-discipline foundation *before* moving any kernel. This is the non-golden,
greenfield groundwork — no compute path changes domain, no precision re-baseline.
Two deliverables: (a) document the two-layer model + boundary-macro rule and promote
`spectral_q15.h` to the canonical Q-domain header, and (b) land a CI contract test
that pins "no float/Q mixing outside the boundary macros."

## What landed

### 1. Canonical Q-domain header documentation (`synth/math/spectral_q15.h`)

A header doc block now states the **two layers** that drive the whole layered design:

- **Storage / transport packing** — boundaries already int16 (PCM out, segment store,
  sine LUT). Q15 here is free/lossless at the boundary; a throughput/bandwidth win.
- **Compute-in-Q15 intermediates** — lossy (~92 dB Q15 SNR ceiling vs the float
  oscillator's -155 dBFS), opt-in per path, float stays default.

It documents the **boundary-macro rule** (every float↔Q conversion goes through a named
macro — `FLOAT_TO_Q15`/`Q15_TO_FLOAT`/`FLOAT_TO_Q31`/`Q31_TO_FLOAT`/`PHASE_RAD_TO_Q15`/
`OMEGA_TO_Q88` here, plus `SPECTRAL_SAMPLE_TO_FLOAT`/`FLOAT_TO_SPECTRAL_SAMPLE` in
`spectral_config.h`) and the **`SPECTRAL_Q_DOMAIN BEGIN/END` region-marker convention**
for pure fixed-point compute blocks. The unconditional Q15 op block
(`spectral_mul_q15`/`spectral_mac_q15`/`spectral_scale_q15`) is now wrapped in the
markers, so the contract test guards a live region from day one (not a vacuous tripwire).

### 2. One real boundary violation removed (`core/spectral_wavetable.c:254`)

The Q15-wavetable→float load did a raw `(float)temp[i] * SPECTRAL_INV_Q15_SCALE`. Routed
through the in-scope boundary macro `SPECTRAL_SAMPLE_TO_FLOAT(temp[i])` — a **byte-for-byte
identical** expansion (`((float)(s) * SPECTRAL_INV_Q15_SCALE)`), so no behavior change,
but the conversion now goes through the named boundary instead of touching the raw scale
constant. This was the only such leak in the engine tree.

### 3. The `q_domain_contract` CTest (test #6)

`cmake/scripts/q_domain_contract.cmake` (a pure `cmake -P` source scan — no compiler, no
Python, runs on every host) wired in via `cmake/targets/q-domain-contract-test.cmake`.
Two rules:

- **RULE 1 — boundary-macro confinement.** The raw scale constants
  (`SPECTRAL_Q15_SCALE`/`SPECTRAL_INV_Q15_SCALE`/`SPECTRAL_Q31_SCALE`/`SPECTRAL_INV_Q31_SCALE`)
  may appear ONLY in the allowlisted definition sites (`spectral_consts.h`,
  `spectral_q15.h`, `spectral_config.h`). Anywhere else = an ad-hoc float↔Q conversion that
  bypassed the named macros → fail.
- **RULE 2 — Q-domain region purity.** No `float`/`double` token inside any
  `// SPECTRAL_Q_DOMAIN BEGIN .. END` region (markers must balance; nesting / orphan-END /
  unterminated all flagged).

The two rules are the two directions of "no mixing": RULE 1 catches a *float* path doing
raw Q conversion; RULE 2 catches a *Q* kernel doing float arithmetic. This is the tripwire
that arms Q2/Q3: when a vectorized Q15 compute kernel lands, wrap it in the markers and it
inherits enforcement.

## Implementation lesson — CMake source scanning

`foreach(line ${content})` over a `\n`→`;`-joined file is **unsafe** for C source: a `[` in
a comment (`radians [0, 2pi)`) opens a CMake bracket-argument and swallows the rest of the
file into one element, `;` splits mid-statement, and blank lines are dropped (corrupting
line numbers). First cut hit exactly this — the scanner's own doc block (which *shows* the
markers as prose) also opened a phantom region. Fixes: (1) walk lines with
`string(FIND/SUBSTRING)` on a quoted string — never re-parse content as a list; (2) anchor
marker detection to comment-led line starts (`^(//|/\*)\s*SPECTRAL_Q_DOMAIN BEGIN`) so prose
mentions are ignored; (3) a fast-path that skips the per-line walk for any file containing
neither a scale constant nor a marker (113/115 files), keeping the test at ~0.03–0.06 s.

## Verification

```text
- 5 production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float).
- ctest 6/6 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity, q_domain_contract).
- Tripwire proven both directions: a temp probe file with a raw
  SPECTRAL_INV_Q15_SCALE *and* a float inside a Q-domain region produced exactly 2
  violations (one per rule); clean again after removal.
- Scanner self-report: "115 files scanned, 1 Q-domain region verified pure, scale
  constants confined to boundary macros."
- wavetable.c change is a byte-identical macro expansion (no golden impact).
```

## Status

Q0 closes. The Q-domain type discipline is documented and enforced by CI *before* any
kernel moves, per the plan's ordering. **Next (Q1)** — storage/transport packing of the
already-int16 boundaries (PCM out, segment store, LUT): pure-win, no precision regression,
measure size/bandwidth. Q1 is non-golden and safe to proceed. **Q2/Q3 remain golden-gated**
(width parameterization of the float L1 kernel; opt-in Q15 compute) — they change
observable output bytes and per the plan's §7 need maintainer sign-off per path before
re-baseline.
