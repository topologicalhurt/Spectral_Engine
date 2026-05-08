# Core audit pass 79: peak window callback neighborhood contract

## Summary

Pass 79 hardens calls into window-specific peak-height callbacks.

The estimator lets window descriptors provide custom peak magnitude-squared
callbacks. Those callbacks receive a local three-bin neighborhood and a sub-bin
offset.

## Bug

The current-frame callback validated the center bin and offset, but did not
validate left/right neighborhood bins before passing them into the callback.

The next-frame callback validated center indirectly but also passed left/right
neighborhood bins directly into the callback.

Built-in callbacks are defensive, but custom or future callbacks should not be
asked to handle NaN/negative neighborhood data. The estimator boundary should
validate callback inputs first.

## Fix

Both current-frame and next-frame peak-height callback paths now validate:

```text
left/right finite and non-negative
center finite and positive
offset finite
```

before invoking the descriptor callback.

## Reviewer Walkthrough

1. The current-frame helper loads left/right bins explicitly.
2. It rejects invalid neighborhood bins before callback dispatch.
3. The next-frame helper applies the same checks to `next_magsq_row`.
4. Gain bounding from Pass 74 remains after callback output.
5. Built-in callback behavior is unchanged for valid inputs.

## Why this is critical

Window descriptors are extension points. The core estimator must define the
input contract before calling them, otherwise invalid STFT neighborhoods can
escape into arbitrary callback code.
