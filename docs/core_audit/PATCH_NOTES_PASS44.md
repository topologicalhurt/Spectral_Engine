# Core audit pass 44: Metal dispatch completion contract

## Summary

Pass 44 fixes a Metal backend dispatch-contract bug.

The Metal backend built command-buffer state, waited for completion, and then
copied from the output buffer without checking whether command creation or
execution actually succeeded.

## Bug

Objective-C messages to `nil` are no-ops. If any of these objects failed to be
created:

```text
paramsBuffer
cmdBuffer
encoder
```

the function could still fall through to output copyback.

The backend also did not check command-buffer completion status after:

```objc
[cmdBuffer waitUntilCompleted];
```

A failed Metal command buffer could therefore leave stale output in the persistent
Metal output buffer and copy it back as if synthesis succeeded.

## Fix

The backend now fails closed if any dispatch object is missing:

```text
paramsBuffer
cmdBuffer
encoder
```

After waiting, it checks:

```objc
[cmdBuffer status] == MTLCommandBufferStatusCompleted
```

before copying back output.

## Reviewer Walkthrough

1. Parameter buffer allocation is checked immediately.
2. Command buffer creation is checked immediately.
3. Compute encoder creation is checked immediately.
4. The command buffer is committed and waited on as before.
5. Completion status is checked before `memcpy()` from the Metal output buffer.
6. Failure paths reuse existing cleanup and zero-output behavior.

## Why this is critical

A GPU backend must not copy stale persistent-buffer contents after dispatch
failure. Command-buffer completion is the boundary between "kernel produced
this output" and "buffer contains unspecified old data."
