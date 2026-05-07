# Core audit pass 46: CUDA async transfer and timing contract

## Summary

Pass 46 hardens CUDA backend asynchronous execution.

Earlier GPU passes fixed parameter packing, event lifetime, Metal completion and
tile fill bounds. CUDA still submitted asynchronous transfers/events without
checking every API boundary before using the resulting device buffers or timing
state.

## Bug

The CUDA path submitted:

```c
cudaMemcpyAsync(...)
cudaEventRecord(...)
cudaStreamSynchronize(...)
cudaEventElapsedTime(...)
```

but several calls were not checked. A failed host-to-device transfer, failed
event record, failed device-to-host copy, or failed stream synchronization could
still fall through as if synthesis succeeded.

That is especially dangerous because device buffers are persistent. If an upload
or copyback fails, the backend may expose stale output or stale segment/tile
data.

## Fix

CUDA dispatch now checks:

```text
SegmentGpu upload
tile ID upload
tile range upload
event record start/stop
kernel launch status
output copyback enqueue
stream synchronization
event elapsed-time query
```

Every failure goes through the common cleanup path, zeros the output buffer, and
returns a GPU error.

## Reviewer Walkthrough

1. Every `cudaMemcpyAsync()` stores and checks its return value.
2. Event recording is checked before and after kernel launch.
3. Kernel launch failure is checked immediately with `cudaGetLastError()`.
4. Device-to-host copyback is checked before synchronizing.
5. `cudaStreamSynchronize()` is checked before reading timing or trusting
   output.
6. Timing query failure is also treated as backend failure.
7. Event cleanup from Pass 43 still handles all success/error paths.

## Why this is critical

CUDA operations are asynchronous. Submitting work is not proof that work was
accepted or completed. A GPU backend must not publish output unless every queue,
kernel, copyback and synchronization boundary succeeded.
