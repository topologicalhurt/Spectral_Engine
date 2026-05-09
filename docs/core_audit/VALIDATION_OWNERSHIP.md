# Validation ownership

Boundary-owned checks stay at:

```text
file shape and size
persistent cache shape and payload
public API input shape
GPU ABI packing
backend command/copy completion
float-to-int or size narrowing
```

Local wrappers are allowed when they improve module readability, but the wrapper
must delegate to `spectral_contracts.h` rather than duplicating field loops.

Do not add a new local finite/range loop unless no canonical helper expresses
the contract.
