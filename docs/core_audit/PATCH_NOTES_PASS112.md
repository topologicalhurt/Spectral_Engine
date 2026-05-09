# Core audit pass 112: tracker candidate batch owner

## Summary

Pass 112 starts Phase E by replacing raw candidate-array/count pairs with a
named owner object.

The fused path previously carried:

```c
uint32_t candidate_batch[SPECTRAL_TRACK_CANDIDATE_BATCH];
size_t candidate_batch_count;
```

as two independent locals. That is small, but it is the core queue state used by
every candidate helper.

## Fix

Adds:

```c
SpectralTrackerCandidateBatch
spectral_tracker_candidate_batch_reset()
```

and wires the fused worker to own one batch object:

```c
SpectralTrackerCandidateBatch candidate_batch = {0};
```

Existing helper APIs are still fed `candidate_batch.ids` and
`&candidate_batch.count` for compatibility. Later passes can collapse the helper
signatures around the owner.

## Why this is critical

The candidate batch is a hot-path data structure. It deserves an explicit owner
before we continue reducing tracker argument lists.
