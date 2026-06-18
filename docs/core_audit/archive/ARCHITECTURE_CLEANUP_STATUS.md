# Architecture cleanup status

## Completed reusable owners

```text
spectral_contracts.h
  finite spans
  Segment payload/synth contracts
  SegmentGpu payload/match contracts
  GPU tile layout contract
  checked counter/time accumulation

spectral_analysis_internal.h / spectral_analysis.c
  analysis shape derivation
  analysis path decision object
  forced full/fused path-mode entry point
  analysis window context
  full-matrix STFT matrix owner

spectral_analysis_fft.c
  single shared FFT frame dispatch
  backend-specific single-frame kernels only

spectral_peak_estimator.c
  double-domain offset clamp helper
  magnitude triplet loader
  shared estimator neighborhood validation

spectral_peak_track_internal.h
  frame-index -> t_hop conversion helper
  frame-context constructor
  candidate batch owner
  worker-stats context

spectral_peak_track.c
  peak-model mutation helper
  stats accumulation helpers
  candidate flush hidden as implementation detail

spectral_omp.h
  effective OpenMP thread-count clamp

spectral_synth_internal.h / spectral_synth_internal.c
  explicit prepared GPU dispatch plan
  host-side GPU segment/tile/params preparation ownership
  one-shot GPU cache lookup encapsulation
```

## Completed phases

```text
Phase C: contract consolidation and alias-wrapper removal
Phase D: explicit prepared GPU dispatch object
Phase E: tracker candidate context object foundation
Phase F: full/fused parity testing seam and harness specification
```

## Current principles

```text
validate at boundaries
derive shape once
use canonical contracts
avoid alias wrappers
delete deprecated internal APIs
prefer owner structs for paired resources
keep backend-specific code backend-specific only
make preparation objects explicit when multiple backends consume the same host-side state
hide implementation-detail long-argument helpers from headers
do not continue the campaign without a real bug, dedup, API-reduction, or behavioral-test payoff
```

## Remaining high-value dedup targets

```text
1. Compiled full/fused parity harness:
   The forced-path seam and spec exist. The next high-value work is a compiled
   behavioral harness that runs deterministic fixtures.

2. GPU tile preprocessing:
   The count/fill algorithm is still dense. Extract a reusable tile-layout
   builder only after parity harnesses are in place.

3. Tracker candidate flow:
   Candidate batch/frame/stats ownership exists. Further signature shortening
   should wait until parity tests protect behavior.

4. ARM/embedded redesign:
   Treat as a separate redesign project, not a continuation of host-kernel cleanup.
```

## Forbidden anti-patterns

```text
pure alias wrapper functions
unchecked fallback wrappers that discard errors
duplicated finite/range loops when spectral_contracts.h has a helper
count-only identity for cached payloads
backend-specific copies of shared dispatch policy
backend-owned host-side GPU dispatch preparation
tests that require stale wrapper names
public/internal headers exposing implementation-detail long-argument helpers
new guard-only passes without boundary-defect evidence
```

## Next recommended phase — LANDED (archived doc; left for history)

```text
Phase G: implement compiled full/fused behavioral parity harness
  → DONE (PASS221): ctest `full_fused_parity` (test_full_fused_parity.c). This was
    the last open phase; the architectural-cleanup campaign is complete.
```
