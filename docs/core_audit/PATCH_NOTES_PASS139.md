# Patch notes — Pass 139: interim ARM-synth behavior oracle

## Problem

The Campaign-2 ARM redesign (ULTRAPLAN Phase A) must prove each refactor is
behavior-preserving, but the compiled parity harness is Phase D (scheduled last)
and the existing `tests/core_math/*` suite is 100% Python string-matching over
source text — it never builds or runs the kernel. There was no oracle to detect
whether an ARM-synth change altered output.

## Change

Add `tests/arm_oracle/oracle.py` (stdlib only): generates deterministic 16-bit
PCM fixtures (bin-centered sine, off-bin sine, two-tone), runs the simulate build
(`synth_arm32_simulation`, the host-side Q15 ARM oracle) over a 6-case
fixture x argument matrix (stretch 1.0 and 1.5), and captures/compares sha256
goldens in `goldens.json`. Modes: `gen`, `capture`, `check`, `diff` (max/RMS PCM
diff of two float WAVs, for passes expected to perturb LSBs e.g. NEON removal).

Determinism was verified before relying on it:

```text
- three consecutive runs are byte-identical;
- cold (cache cleared) == warm (cache hit): the spectral_seg_cache round-trip is
  lossless and synthesis re-runs every invocation, so the oracle faithfully
  reflects ARM-synth changes;
- output is 32-bit float WAV (fmt=3); parsed directly for the diff tool.
```

## Scope / why minimal

No kernel code changed — this is a test/harness addition (a valid pass per the
closure criteria). It is an *interim*, environment-specific regression anchor
(hashes depend on this toolchain's sim binary; re-capture per environment) and is
explicitly superseded by the Phase D compiled tolerance harness.

`goldens.json` and `oracle.py` are committed; `fixtures/*.wav` are git-ignored
(`*.wav`) and regenerated via `oracle.py gen`.

## Incidental observations (logged for later phases)

```text
- A 0-segment input (e.g. silence) is rejected by the CLI as "input error"
  before synthesis; excluded from the matrix (no ARM-synth path). Possible minor
  CLI-semantics nit for Phase C: 0 segments classified as input error rather
  than a distinct no-op status.
```
