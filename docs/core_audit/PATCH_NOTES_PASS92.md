# Core audit pass 92: synthesis preflight consolidation and legacy API removal

## Summary

Pass 92 removes legacy synthesis preflight wiring and pure compatibility aliases.

This is an architecture cleanup pass. It does not add new defensive checks. It
pays down tech debt introduced by earlier hardening by making the canonical
preflight path the only internal synth validation path.

## Problems

### 1. Output byte count was recomputed

`synth_preflight_common()` computed:

```c
preflight_out_bytes = out_len * elem_size
```

with checked arithmetic, then called:

```c
synth_validate_inputs(...)
```

which recomputed the same byte product.

That violates the current kernel guideline: derive shape once and pass/reuse the
checked result.

### 2. Legacy validate API remained public inside the internal header

The internal header still exposed:

```c
SynthValidateResult
synth_validate_inputs()
SYNTH_VALIDATE_FLOAT
SYNTH_VALIDATE_NATIVE
```

but backend code now uses `synth_preflight_float()` /
`synth_preflight_native()`.

Those older macros/functions are deprecated internal API.

### 3. Synthesis segment validation had a duplicate local loop

`synth_segment_payload_valid()` duplicated the canonical contract now provided
by:

```c
spectral_segment_array_valid_for_synth()
```

### 4. Unchecked GPU pack alias remained

The header still exposed:

```c
gpu_synth_params_pack(...)
```

which silently discards failure from `gpu_synth_params_pack_checked()`.

Metal/CUDA already use the checked packer, so the unchecked alias is dead and
dangerous.

## Fix

Pass 92:

```text
inlines early-exit handling into synth_preflight_common()
reuses preflight_out_bytes everywhere in preflight
removes synth_validate_inputs()
removes SynthValidateResult and SYNTH_VALIDATE_* macros
removes synth_segment_payload_valid()
removes gpu_synth_params_pack()
keeps gpu_synth_params_pack_checked() as the only GPU ABI packer
updates docs/audit/tests to enforce this architecture
```

## Why this is critical

Synthesis preflight is the front door to every backend. There should be exactly
one internal path that establishes the output/segment/parameter contract before
CPU, native, Metal or CUDA code runs.
