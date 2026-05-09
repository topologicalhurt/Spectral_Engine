# Kernel patching guidelines

These rules apply to every future core/kernel pass.

## 1. Do not add alias wrappers

Avoid functions whose body is only:

```c
return other_function(...);
```

Use the canonical function directly at the call site.

Allowed exceptions:

```text
public ABI compatibility
macro-selected backend dispatch
type-adapting wrappers that do real validation or conversion
callbacks required by an external interface
```

A local wrapper is not allowed merely to preserve an old pass name or make a
test easier.

## 2. Validate at boundaries, not everywhere

Boundary validation belongs at:

```text
public API ingress
file/cache load
file/cache store
GPU ABI packing
backend command/copy/sync completion
float/integer narrowing
extension-point callback dispatch
```

Hot loops should consume validated state and only check derived quantities they
create themselves.

## 3. Centralize reusable contracts

Use canonical helpers from:

```text
spectral_engine/core/spectral_contracts.h
```

before adding any new finite/range loop.

Current canonical helpers include:

```text
spectral_f32_span_finite
spectral_f32_span_finite_nonnegative
spectral_segment_payload_valid
spectral_segment_valid_for_synth
spectral_segment_array_payload_valid
spectral_segment_array_valid_for_synth
spectral_segment_gpu_payload_valid
spectral_segment_gpu_array_matches_segments
spectral_gpu_tile_layout_words_valid
```

## 4. Keep derived-state constructors canonical

Do not recompute derived backend state in multiple places. Use or create one
constructor/packer:

```text
make_synth_params
gpu_synth_params_pack_checked
spectral_tracker_derive_create_scalars
spectral_fft_resources_set_magsq_scales
```

## 5. Remove legacy code when a contract supersedes it

When a canonical contract is introduced:

```text
replace call sites
delete pure wrappers
update old static tests
update audit expectations
document the ownership change
```

Do not leave deprecated wrappers around “just in case.”

## 6. Tests should enforce contracts, not stale implementation names

Static tests may assert dangerous expressions are absent. They should not force
old alias wrapper names to remain.

Good:

```text
no raw float-to-int cast
no count-only GPU cache reuse
canonical Segment contract is used before synthesis
```

Bad:

```text
function named seg_cache_segments_valid must exist
```

## 7. Kernel patches must be strategically minimal

Each pass must either:

```text
remove duplication
fix a real contract bug
simplify wiring
delete legacy/deprecated code
improve test/audit ownership
```

A patch that only adds more local defensive checks is rejected unless it closes a
newly identified boundary bug and updates the canonical contract layer.
