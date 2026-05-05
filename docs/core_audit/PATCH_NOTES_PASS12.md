# Core audit pass 12: estimator contract hardening

## Why this pass exists

Pass 11 created a real estimator subsystem. That was the right architectural
move, but it also introduced a new distinction:

```text
spectral_peak_estimate()           public-safe path
spectral_peak_estimate_validated() tracker hot path
```

The word "validated" must not become a license to skip invariants that the
tracker did not actually prove.

## Finding

`spectral_peak_estimate_validated()` was skipping some checks under the
assumption that tracker candidate validation had already accepted the local
magnitude neighborhood.

The tracker does prove a local current-frame and next-frame magnitude
neighborhood before emission, but it does not prove every scalar field in
`SpectralPeakEstimateInput` for every future caller of the validated API. It
also does not prove phase finiteness for complex-estimator reconstruction.

## Changes

- Complex coefficient reconstruction always checks magnitude-squared and phase.
- Log-power and magnitude parabolic paths always reject non-finite or negative
  three-bin neighborhoods before calling descriptor interpolation callbacks.
- `spectral_peak_estimate_impl()` now unconditionally validates:
  - `curr_magsq`
  - `next_max_magsq`
  - `freq_step_omega`
  - `freq_step_df`
  - `inv_hop`
- Added a C-backed Python test proving the validated API rejects NaNs/negative
  triplets and invalid scalar context.

## Intuition

A fast path can skip repeated work only when the earlier stage actually proved
the same property. Candidate validation proves "this is a local magnitude peak
above threshold." It does not prove "every future estimator input field is
finite and safe." The estimator module is now a reusable kernel, so those
contracts must live inside the estimator too.
