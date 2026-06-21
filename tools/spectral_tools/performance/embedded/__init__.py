"""Cortex-M7 performance measurement stack (M7_PERF_MODEL_PLAN).

Layered, with measured and modeled quantities kept distinct:

- ``codegen``  — Layer 0/2: real-TU instruction census [measured: codegen] and
  per-loop steady-state cycles via llvm-mca CortexM7Model [modeled].
- ``counts``   — Layer 1: exact dynamic instruction/byte counts of the real
  kernel under qemu-system-arm mps2-an500 [measured: qemu-tcg].
- ``fixture``  — single source of truth for the deterministic workload; emits
  the C header the QEMU runner compiles against.
- ``toolchain`` — discovery of the cross/analysis toolchain with the Daisy
  production flags.

CLI entry: ``python -m spectral_tools.testing.benchmark_workflow m7-census|m7-counts``.
"""

__all__ = [
    "codegen",
    "counts",
    "fixture",
    "toolchain",
]
