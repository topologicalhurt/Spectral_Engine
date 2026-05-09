# Phase C: contract consolidation and deduplication

Phase C converts previous local hardening into maintainable kernel architecture.

It introduces:

```text
spectral_engine/core/spectral_contracts.h
docs/core_audit/CORE_CONTRACTS.md
docs/core_audit/VALIDATION_OWNERSHIP.md
tests/core_math/test_core_phase_c_contract_consolidation.py
```

It consolidates duplicated logic for:

```text
finite float spans
Segment payloads
synthesis SegmentArray payloads
SegmentGpu payload/match checks
GPU tile layouts
```

Local domain names remain as thin wrappers so older tests and readable module
semantics survive, but duplicated validation bodies are removed.
