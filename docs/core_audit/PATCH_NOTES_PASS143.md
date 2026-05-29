# Patch notes — Pass 143: CTest correctness harness over the REAL ARM process

## What

`tests/arm_core/test_arm32_process.c` + a CTest target (`arm_core_test`) that
build the actual embedded synth (`spectral_arm32_init/load/process`,
`synth_core_m7`) on the host with `SPECTRAL_ARM_M7` forced (portable intrinsic
fallbacks, Pass 142) and assert behavioral correctness:

```text
- no segments      -> process renders 0 samples and zeros the output block;
- single tone      -> dominant output frequency tracks the NOMINAL input
                      frequency (Goertzel scan; non-circular — compared to the
                      input Hz, not the synth's own scaling), output audible and
                      non-clipping.
```

This is the correctness half of ULTRAPLAN A1b. It replaces the interim sim-hash
oracle for correctness, which was vacuous for the embedded code: the sim
(`synth_arm32_simulation`) reimplements synthesis and never runs
`spectral_arm32_process`.

```text
Run: cmake --build build --target arm_core_test && ctest --test-dir build -R arm32
```

## Result

On its first run the harness caught a real frequency-scaling bug in the embedded
hot path (fixed in Pass 144). After the fix: 1/1 passing.

## Follow-ups (logged, not yet done)

```text
- Amplitude: a single 0.5-amp segment renders at ~0.25 (-6 dB). The accumulator
  holds Q30 products (Q15*Q15) but spectral_q31_to_q15_* shifts by 16 (as if Q31).
  Deliberate 1-bit polyphony headroom, or an off-by-one vs the desktop float
  backend? Unresolved -> the harness asserts only audible+non-clipping. Resolve by
  comparing absolute level against the desktop float backend.
- validate_segment_data is still orphaned (-Wunused-function): load-path boundary
  validation is not wired in. Now that the harness exercises load, wire it in with
  a rejection test for invalid input (next pass).
```
