# Patch notes — Pass 140: remove hand-written ARM NEON from spectral_q15.c

## Problem

`spectral_q15.c` was the only kernel file using raw architecture intrinsics
(`<arm_neon.h>`: `vld1q_s32` / `vqshrn_n_s32` / `vmull_s16`) for the bulk
Q31->Q15 conversion, gated on `__ARM_NEON`. The rest of the engine already
abstracts SIMD through SIMDe (`oscillator_simd.c`, `spectral_vector_ops.c`: write
SSE2 -> NEON/SSE/scalar) and CMSIS on embedded. The hand-NEON path:

```text
- only ever compiled on desktop ARM; Cortex-M (the embedded target) has no NEON,
  so it was dead for the target it nominally served;
- is not on the embedded hot path (final output normalization, O(n) per block);
- duplicated, in a non-portable form, logic that has an exact scalar reference.
```

## Change

Remove the `__ARM_NEON` path; `spectral_q31_to_q15_bulk` / `_scaled` are now the
single portable scalar implementations. Per AI_CANON 11 the campaign keeps no
unproven SIMD; if this conversion is ever shown hot, re-add via SIMDe behind a
benchmark, not raw arch intrinsics.

## Verification

Using the interim oracle (Pass 139, `tests/arm_oracle`): the NEON build and the
portable build produce byte-identical PCM — max abs sample diff = 0, rms = 0 —
over all fixtures. The two builds' output WAVs differ in exactly one byte: the
WAV PEAK-chunk `timeStamp` (wall-clock), which is not audio.

That one byte exposed a flaw in the Pass-139 oracle: it hashed the whole WAV, so
the PEAK `timeStamp` made the hash time-sensitive (a `check` run a second after
`capture` would spuriously fail). Hardened the oracle to hash the `data` chunk
payload (audio) only; `check` now reruns the sim with fresh timestamps and still
matches. `goldens.json` re-baselined to the PCM hash.

## Arch-separation finding (the "separate arches into files" item)

Arch handling is already factored: SIMDe (desktop portable) + CMSIS (embedded) +
`backends/arm` for ARM-specific synth. `spectral_q15.c`'s hand-NEON was the sole
outlier; with it gone, no kernel file carries raw per-arch intrinsics. Splitting
`oscillator_simd.c`'s CMSIS/SIMDe branches into separate files was considered and
not done — it would fight SIMDe's write-once design without reducing real
complexity. (Open to it if a per-arch file layout is specifically wanted.)

## Incidental observation

The WAV PEAK chunk embeds a wall-clock `timeStamp`, so audio output files are not
byte-reproducible across time (the audio is). Conventional for the PEAK chunk;
noted for future build-reproducibility/caching work, not a defect.
