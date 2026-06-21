# ADR-0002: Performance-Measurement Pipeline and Instrument Matrix

- Status: Accepted
- Date: 2026-06-10
- Decision makers: maintainer (topologicalhurt)

## Context

Performance numbers inform core design decisions and must stand in for
hardware that is not present (Cortex-M7/Daisy). Requirements set by the
maintainer: one cohesive pipeline (reproducible build → run under a
wrapper/observer → parse → interpret); the correct tool per build target; no
hand-rolled solutions where industry-standard ones exist; and Python must not
duplicate C logic or constants.

## Decisions

### 1. The pipeline shape

Every measurement follows: **build reproducibly (cmake, recorded
args/git-head) → run under an instrument → parse to a typed result →
interpret in a report**. The orchestration lives in
`spectral_tools.testing.benchmark_workflow` (one CLI; CMake targets wrap it).
`spectral_tools.performance.matrix` is the SSOT mapping each build target to
its instruments, with runtime availability probes (`measure --list`).

### 2. Timing instrument: in-process markers, never debuggers

Stage timing uses the engine's `SPECTRAL_STAGE_BEGIN/END` markers
(`CLOCK_MONOTONIC` ns, `spectral_log.h`). Measured on the dev machine: a
marker costs ~600 ns; an lldb auto-continue breakpoint stop costs
~3.2–3.5 ms — a ~5,700× penalty per event, all-stopping the OMP pool, and gdb
has no native arm64-macOS support at all. On Linux, sourced overhead is
~95 µs per gdb breakpoint encounter vs effectively-free `perf stat` counting.
**Debuggers are therefore never a timing instrument.** Their sanctioned
roles: state inspection on anomalous runs, postmortems, and (future) QEMU
gdbstub stage segmentation — sound there because guest virtual time cannot
advance while the VM is stopped, so segmentation is distortion-free.

### 3. Counters per platform

Linux: `perf stat` (exists, `perf_profile.py`). macOS headless: none today
(xctrace requires full Xcode, absent; kperf private; dtrace SIP-blocked) —
markers + BSD `time -l` RSS carry the row. Cross-M7: QEMU TCG plugin counts
[measured] + llvm-mca CortexM7Model [modeled]. On-target Daisy (future):
DWT.CYCCNT is the calibration anchor when hardware lands.

### 4. The C-truth rule (porting/duplication injunction)

Python derives facts from the C/CMake artifacts; it does not restate them:

- Build flags are parsed at runtime from `options.cmake`
  (`toolchain.daisy_target_flags`); a failed parse raises — no fallback copy.
- Symbol addresses come from `nm` on the built ELF; counts from the binary
  under QEMU; censuses from compiled assembly.
- Where Python must originate data the C side consumes (the workload
  fixture), it is **generated into C** (`spectral_fixture_generated.h`) with
  a content digest carried through every report — one direction, no parallel
  versions.
- The only sanctioned duplication is an *independent verification
  implementation* (e.g. `testing/resource_hash_reference.py`), which exists
  to disagree with the C, is labeled as such, and is parity-tested.

### 5. Provenance tagging

Every reported number is tagged `measured: <instrument>` or
`modeled: <model>`; measured and modeled quantities are never blended into
one number. QEMU counts are counts — never cycles (fidelity contract in
`docs/core_audit/M7_PERF_MODEL_PLAN.md`).

### 6. Real newlib, no stub headers

The embedded rigs compile against a real newlib toolchain (xPack GNU Arm
Embedded GCC, sha-pinned `m7-bootstrap` download into `tools/toolchains/`,
gitignored). The previous freestanding stub headers are deleted: measured
DSP census was identical, and the runner now links `-lc_nano` so libc traffic
in counts matches what Daisy firmware (nano.specs) executes. The brew
bare-metal gcc fails the newlib probe and is rejected with an actionable
error.

### 7. Orchestration stays in-repo

Survey found no established open-source framework for a mixed
embedded+desktop build→instrument→parse matrix (LNT is llvm-test-suite- and
server-shaped; Google Benchmark is in-process; asv is Python-centric). A thin
in-repo matrix is standard practice (cf. Zephyr twister). Idea adopted from
LNT: one JSON report schema (`suite`/`context`/`tests`) shared by every row.

## Consequences

- One entry point and one declarative matrix; adding an instrument means one
  registry entry + probe, not a new script.
- Measurements are reproducible by construction (pinned toolchain, fixture
  digests, duplicate-run count verification) and honest by construction
  (provenance tags, loud failures when a tool or contract is missing).
- Debugger-based timing proposals can be declined by pointing at the
  measured numbers above.
