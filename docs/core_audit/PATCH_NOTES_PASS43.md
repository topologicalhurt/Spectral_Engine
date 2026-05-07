# Core audit pass 43: CUDA event lifecycle contract

## Summary

Pass 43 fixes a CUDA backend resource-lifecycle bug.

The CUDA backend creates timing events around kernel dispatch. Several error
paths after event creation jumped to the common cleanup label before the old
event-destroy calls were reached.

## Bug

The old structure was:

```c
cudaEvent_t ev_start, ev_stop;
cudaEventCreate(&ev_start);
cudaEventCreate(&ev_stop);

...
if (kernel_err != cudaSuccess) goto cleanup;

cudaEventDestroy(ev_start);
cudaEventDestroy(ev_stop);

cleanup:
    ...
```

If kernel error handling or later cleanup-bound paths were taken, the events
were leaked. The code also ignored event creation failures.

## Fix

CUDA event handles now live in the same scope as the cleanup label:

```c
cudaEvent_t ev_start = NULL;
cudaEvent_t ev_stop = NULL;
```

Event creation is checked. The common cleanup label destroys any event that was
successfully created.

## Reviewer Walkthrough

1. Event handles are initialized before any `goto cleanup` path can be taken.
2. Event creation failures are classified as GPU init failures.
3. The old pre-cleanup unconditional destroy block is removed.
4. Cleanup destroys `ev_start` and `ev_stop` if non-null.
5. All success and error paths share the same event cleanup policy.

## Why this is critical

Backend dispatch resources must have one cleanup policy. Timing events are not
large, but leaks in a per-render GPU path become persistent device-resource
leaks and can destabilize longer sessions.
