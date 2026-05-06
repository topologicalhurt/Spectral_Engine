# Core audit pass 6: FFT resource ownership hardening

## Summary

Pass 6 fixes FFT resource ownership around partial allocation failure.

## Findings

`spectral_fft_resources_alloc()` allocated pointer arrays with ordinary
allocation and returned early on failures. If one pointer-array allocation
failed after another had succeeded, `spectral_fft_resources_free()` could
inspect uninitialized pointer-array entries. The allocator also returned
directly from inner allocation failures, depending on every caller to run
cleanup.

For a kernel-style library, allocation ownership must be local and
deterministic.

## Changes

- Pointer arrays are zero-initialized with `spectral_calloc_array`.
- Allocation failures use one `goto fail` cleanup path.
- The cleanup path calls `spectral_fft_resources_free(res)`.
- `spectral_fft_resources_free()` is null-safe.
- `spectral_fft_resources_free()` zeroes the resource struct after releasing
  owned memory.

## Validation

Run:

```sh
python3 tests/core_math/test_core_pass6_static.py
python3 tools/core_audit/core_static_audit.py .
git diff --check
make clean && make configure CMAKE_BUILD_TYPE=Debug
make desktop
```
