# Core audit pass 39: synthesis derived-parameter finiteness contract

## Summary

Pass 39 fixes a synthesis parameter-domain bug.

Earlier passes made `stretch` validation centralized and required it to be
finite, positive, and below `SPECTRAL_MAX_STRETCH`. That is necessary but not
sufficient: every backend consumes derived scalars, not only the raw input.

## Bug

`make_synth_params()` derived:

```c
inv_stretch = 1.0f / stretch
inv_stretch_sq = 1.0f / (stretch * stretch)
pitch_factor = SPECTRAL_PITCH_FACTOR(pitch)
```

after `synth_validate_params()` accepted any finite positive stretch.

For tiny positive `stretch`, `stretch * stretch` can underflow to zero in
`float`. That makes:

```text
inv_stretch_sq = Inf
```

even though the original `stretch` passed validation.

That poisoned `SynthParams`, which are shared by CPU, GPU fallback, native, and
simulation synthesis paths.

## Fix

Parameter validation now derives the actual backend scalars in one helper:

```c
synth_derive_param_scalars(...)
```

The helper validates:

```text
raw stretch/pitch domain
stretch * stretch is finite and positive
1 / stretch is finite and positive
1 / (stretch * stretch) is finite and positive
pitch factor is finite and positive
```

`make_synth_params()` now reuses the checked derived values instead of
recomputing them independently.

## Reviewer Walkthrough

1. `synth_validate_params()` now delegates to `synth_derive_param_scalars()`.
2. The helper first applies the existing raw stretch/pitch bounds.
3. It then computes `stretch_sq = stretch * stretch`.
4. If `stretch_sq` underflows to zero or becomes non-finite, validation fails.
5. It computes `inv_stretch`, `inv_stretch_sq`, and `pitch_factor`.
6. All three derived values must be finite and positive.
7. `make_synth_params()` receives the already-checked derived values and stores
   those into `SynthParams`.
8. Existing preflight behavior still zeroes the output buffer on parameter
   failure.

## Why this is critical

The synthesis kernel contract is not "inputs are finite"; it is "the scalars
consumed by the hot loops are finite." A tiny but positive stretch can be a
mathematically invalid backend request if it produces infinite derived state.
Letting that reach oscillators makes the backend behavior undefined from the
engine contract's point of view.
