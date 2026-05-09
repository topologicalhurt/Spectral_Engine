# Core contracts

Validation belongs at boundaries. Hot loops should consume already validated
state and only check derived hot-loop quantities that are created inside the hot
loop.

## Canonical reusable contracts

```text
spectral_f32_span_finite
spectral_segment_payload_valid
spectral_segment_valid_for_synth
spectral_segment_array_payload_valid
spectral_segment_array_valid_for_synth
spectral_segment_gpu_matches_segment
spectral_segment_gpu_array_matches_segments
spectral_gpu_tile_layout_words_valid
```

## Segment payload contract

```text
start  finite, >= 0
length finite, >= 0
phase  finite
omega  finite, >= 0
df     finite
amp    finite
da     finite
width  finite
```

## Synthesis segment contract

```text
payload segment contract
length > 0
amp >= 0
```

## Tile layout contract

```text
range[i].start == running_refs
running_refs + range[i].count does not overflow
final running_refs == total_refs
every tile_segment_ids[j] < segment_count
```
