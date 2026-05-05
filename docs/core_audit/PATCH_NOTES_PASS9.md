# Core audit pass 9: window metric validity flags

## Intuition

`SpectralWindowMetrics.flags` are a contract: if a flag says a calibration is
valid, downstream code is allowed to trust the corresponding scale. That means
the flag must describe the *derived scale*, not merely the window sum used to
compute it.

A caller-provided future window can have a finite positive but extremely tiny
sum. In that case `2 / sum(window)` or its square can overflow. The window sum
is technically valid, but the calibration is not usable.

## Fix

Pass 9 computes endpoint and interior-bin amplitude and magnitude-squared
scales in double precision, casts to float, and sets each validity flag only if
both the amplitude scale and the magnitude-squared scale are finite and
positive.

Invalid scale fields remain at the neutral fallback value `1.0f`, but their
validity flags stay clear.

## Documentation correction

The pass-8 notes now match the current code:

- descriptor-based window registry;
- separate endpoint/interior-bin scales;
- frame maxima recomputed from scaled interior trackable bins;
- no single uniform `magsq_scale` contract.
