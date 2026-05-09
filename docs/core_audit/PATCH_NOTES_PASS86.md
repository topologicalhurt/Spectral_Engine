# Core audit pass 86: synthesis segment payload preflight contract

## Summary

Pass 86 hardens the synthesis producer/consumer boundary.

The tracker and cache now validate segment payloads, but public synthesis can
still be called with a caller-provided `SegmentArray`. CPU synthesis tends to
skip invalid segments through `segment_loop_params_init()`, while GPU synthesis
can pack and upload invalid fields before shader dispatch.

## Bug

`Synth_preflight_common()` validated output shape and stretch/pitch, but did not
validate the actual segment payload before backend execution.

That means invalid caller-provided segments could reach:

```text
CPU/native callback loops
SegmentGpu packing
Metal/CUDA shader dispatch
```

even though downstream code increasingly assumes segment payloads are valid DSP
state.

## Fix

Synthesis preflight now validates every segment before creating `SynthParams`:

```text
finite non-negative start
finite positive length
finite phase
finite non-negative omega
finite df/da
finite non-negative amplitude
finite width
```

Invalid segment arrays zero the output buffer and return `SPECTRAL_ERR_PARAM`.

## Reviewer Walkthrough

1. Benign empty segment arrays still produce silence through the existing early
   exit path.
2. Non-empty segment arrays must satisfy the payload contract.
3. The check runs before backend-specific CPU/GPU code.
4. The same preflight covers float, native, Metal and CUDA paths.

## Why this is critical

Synthesis is a kernel boundary. It should not depend on the cache or tracker
being the only possible segment producers. Public/caller-provided segment arrays
must be validated before any backend consumes them.
