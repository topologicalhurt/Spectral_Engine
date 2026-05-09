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
```

## Remaining high-value dedup targets

```text
1. GPU tile preprocessing:
   The count/fill algorithm is still dense. Consider extracting a reusable
   tile-layout builder with an explicit scratch object.

2. Tracker candidate flow:
   Candidate batch/frame/stats ownership now exists, but queue/handle/batch
   helpers still have long internal signatures. Further shortening should be
   done only after behavioral parity tests are in place.

3. Full/fused parity:
   Add behavioral parity tests around analysis output, not only static structure
   tests.

4. Peak estimator algebra:
   Extract more helpers only when doing so reduces code size without hiding the
   formulas.
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
```

## Next recommended phase

```text
Phase F: full/fused parity harness and behavioral tests
```
