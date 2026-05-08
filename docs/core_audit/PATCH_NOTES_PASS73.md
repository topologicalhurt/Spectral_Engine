# Core audit pass 73: tracker candidate-batch bounds contract

## Summary

Pass 73 hardens tracker candidate-batch handling.

Peak scan queues candidate bin indices into a fixed stack batch:

```c
uint32_t candidate_batch[SPECTRAL_TRACK_CANDIDATE_BATCH]
```

The queue/flush helpers are internal, but they are the boundary between SIMD
scan masks and ALU-heavy segment emission.

## Bug

`Spectral_tracker_flush_candidate_batch()` dereferenced
`candidate_batch_count` before checking it, and did not validate that the count
was within the fixed batch capacity.

`Spectral_tracker_queue_candidate()` wrote:

```c
candidate_batch[(*candidate_batch_count)++] = candidate;
```

without first proving the count was below capacity.

If a future call path corrupted the count or called the helper incorrectly, the
batch boundary could be overrun before any flush occurred.

## Fix

Flush now validates:

```text
tracker
tid domain
candidate count pointer
local segment counter
batch pointer when count > 0
row/next/phase pointers
candidate_count <= SPECTRAL_TRACK_CANDIDATE_BATCH
```

Queue now flushes a full batch before writing, then verifies capacity again
before appending the new candidate.

## Reviewer Walkthrough

1. Zero-count flush remains a no-op success.
2. Nonzero flush requires a valid candidate buffer and rows.
3. Oversized candidate counts record overflow and reset the count.
4. Queue checks whether the batch is already full before writing.
5. Queue writes only after proving there is capacity.

## Why this is critical

A fixed-size candidate batch is a kernel boundary. SIMD scans may be fast, but
the transition from bitmask to candidate array must never write outside the
batch capacity.
