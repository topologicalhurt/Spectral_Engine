# Core audit pass 75: peak phase-advance finite-product contract

## Summary

Pass 75 hardens the phase-consistency diagnostic used by peak estimation.

Phase-policy work made the estimator track the relationship between magnitude
peak motion and adjacent-frame phase advance. That diagnostic still performed
several float-domain products before proving they were representable.

## Bug

The old phase-advance path computed:

```c
expected_center = (float)bin * freq_step_omega * hop_float
denom = freq_step_omega * hop_float
phase_error = wrap(phase_delta - model_omega * hop_float)
```

For large but finite bin/hop inputs, these float products can overflow to Inf or
lose the representability contract before phase wrapping.

## Fix

The estimator now derives these products in `double`:

```text
expected_center_d
residual_arg_d
denom_d
phase_omega_d
phase_error_arg_d
```

It checks each value is finite and representable as `float` before narrowing
into the float-domain phase wrapper and public estimate fields.

## Reviewer Walkthrough

1. Input pointers and scalar domains are still validated first.
2. `input->bin` is checked against the float representability domain.
3. Phase delta must be finite.
4. Expected center phase and denominator are computed in double.
5. Residual and phase-error wrapper arguments must fit in `float`.
6. Phase omega is computed in double and narrowed only after checking.
7. Outputs are assigned only after all finite-product checks pass.

## Why this is critical

Phase diagnostics are coupled to the estimator policy. They must not report
consistency or phase-derived omega from overflowed intermediate products.
