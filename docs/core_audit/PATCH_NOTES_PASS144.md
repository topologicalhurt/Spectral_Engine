# Patch notes — Pass 144: fix ARM frequency scaling (rendered ~sr/2pi too low)

## Problem

`spectral_arm32_init` computed `freq_inc_scale_q24 = 2^24 / sample_rate`, but
`freq_q88` stores `omega` (rad/sample) * 256 — not Hz. The phase accumulator is
sample-rate-independent (one cycle = 2^32), so the per-sample increment must be
`(omega/2pi) * 2^32 = freq_q88 * 2^24/(2pi)`. The `2^24/sample_rate` divisor
treated `freq_q88` as Hz and rendered every partial about `sample_rate/(2pi)`
≈ 7019x too low — a 1 kHz tone came out near DC.

This was invisible until Pass 143 because the host sim (`synth_arm32_simulation`)
reimplements synthesis from the float `omega` directly and never used `freq_q88`
or `freq_inc_scale_q24`. The Pass-141 "freq verified (430->430)" measurement was
of the sim, not of the real `spectral_arm32_process`.

## Fix

```c
ctx->freq_inc_scale_q24 = (uint32_t)((double)(1u << 24) / SPECTRAL_TWO_PI + 0.5);
```

A sample-rate-independent constant (~2670177). `freq_q88 * scale = omega*2^32/(2pi)`.

## Verification

`tests/arm_core` (the real process on host): a nominal 1000 Hz segment now renders
with dominant frequency 990 Hz (correct within Q8.8 omega quantization); it was
<= 50 Hz (near DC) before. The sim oracle is unchanged (the sim never used this
path). Real-ARM cross builds preserve behavior (scale formula is sr-independent);
verify on-target when a toolchain is available.
