# Core audit pass 113: tracker frame-context constructor

## Summary

Pass 113 centralizes construction of `SpectralFrameContext`.

Fused analysis manually assigned:

```text
row
next_row
phase_row
next_phase_row
t_hop
threshsq
can_start_new
```

after separately deriving `t_hop`.

## Fix

Adds:

```c
spectral_tracker_frame_context_init()
```

The helper derives `t_hop` with the existing frame-time helper and then fills the
frame context. The fused path now calls that constructor.

## Why this is critical

`SpectralFrameContext` is the data contract passed into candidate scanning.
Manual field-by-field construction duplicates wiring and makes future fields
easy to forget.
