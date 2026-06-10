# Master plan closure criteria

This document defines when the current audit/refactor campaign should stop adding new passes.

## Close the campaign when these are true

```text
1. Canonical contracts exist for shared invariants.
2. Backends use explicit preparation objects where host-side state is shared.
3. Full/fused analysis have a forced-path test seam.
4. A documented parity harness exists.
5. Architecture status documents remaining risks.
6. New proposed patches fix real defects or remove measurable duplication.
```

## Do not continue with passes for

```text
minor spelling
pure commentary
new local guard checks with no boundary defect
static tests that only memorialize stale names
wrappers that call one other function
micro-refactors in ARM/embedded paths scheduled for redesign
```

## Require justification for every future pass

Every future pass must state one of:

```text
real bug
real ownership cleanup
real API surface reduction
real test/harness improvement
real performance-path simplification
```

If it cannot, do not patch.

## Current remaining large items

```text
compiled full/fused parity harness
GPU tile layout builder object
possibly tracker candidate context signature reduction after parity tests
ARM/embedded redesign as a separate project
```
