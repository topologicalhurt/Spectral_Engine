# Core audit pass 19: estimator safety and audit bump

## Summary

Pass 19 closes the remaining peak-estimator review findings before the docs
refresh. It keeps default DSP behavior unchanged and adds guardrails around the
public estimator wrapper, benchmark accounting and raw single-shot descriptor
coverage.

## Changes

- `spectral_peak_estimate()` rejects malformed `best_next_bin` values before
  any `df` subtraction can occur. The public-safe path now rejects negative
  values, bins outside the `[bin - 1, bin + 1]` next-frame search contract, and
  bins outside the representable/indexable frequency range.
- `peak_estimator_bench.c` precomputes reference offsets and amplitude targets
  outside the timed estimator loop. Reported `p50_ns` and `p95_ns` now measure
  estimator cost rather than reference-formula overhead.
- The Pass 11 contract harness calls
  `spectral_track_peaks_with_window_descriptor()` with a custom descriptor,
  verifies that emitted `omega` changes, verifies center-amplitude fallback for
  interpolation-only descriptors, and verifies that invalid explicit
  descriptors fail closed instead of silently tracking as Hann.

## Validation

Run:

```sh
pytest -q tests/core_math
python3 tools/core_audit/core_static_audit.py
git diff --check
cmake -S . -B /tmp/spectral-review-build -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/spectral-review-build -j2
```
