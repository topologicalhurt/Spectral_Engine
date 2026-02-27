# TODO's:

## Immediate (macro-level):

1. Complete testsuite covering spectral_engine
2. Pass testsuite once completed (achieve first stable minor version)
3. Implement realtime file streaming to avoid loading into memory at once & enable realtime capability
4. Reduce .bin file (or segment file) size by using compression techniques & storing the minimum recoverable amount
5. Complete testsuite covering daisy_seed & pass
6. Refactor, consolidate, introduce de-facto design patterns (achieve second stable minor version)
7. Expose more real-time extrinsics, use asm explicitly & achieve atomic correctness for critical parts of embedded pipeline
8. Implement DMA
9. Achieve or exceed performance targets & ensure operation within memory / performance budget
10. Ensure complete correctness of program up to that point (achieve third stable minor version)

11. Implement literature-backed optional processing stages (full algorithms, not stubs)
    - Serra & Smith (1990) — deterministic + stochastic decomposition stage
        - [ ] Add residual model path and deterministic/residual split policy
        - [ ] Add validation metrics (segment reduction, spectral error, perceptual checks)
        - [ ] Add CPU/embedded-friendly fallback mode and profile impact
    - Johnston (1988) — psychoacoustic masking stage
        - [ ] Implement masking threshold model and configurable margin
        - [ ] Prune / down-weight inaudible components with hard safety bounds
        - [ ] Add quality guardrails for transients / speech intelligibility
    - Adaptive track-density policy (informed by McAulay & Quatieri, 1986)
        - [ ] Implement segment budget / keep-ratio policy with deterministic ordering
        - [ ] Add mode presets (transparent / balanced / embedded_turbo)
        - [ ] Ensure baseline harmonic-bin tracking logic remains unchanged (always-on base layer)

## Immediate (micro-level):

1. Trie file system implementation using linked list
    - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8971600 

## Ongoing:

- Improving throughput / memory constraints:
    - Double buffering CPU & GPU
    - Ensure segment binaries or files in VRAM while threads spinning up
    - Support batching the memory or queuing instructions if we can't keep up in realtime (this is a requirement for embedded targets!)

- Improving performance:

    - Introducing FPU / float support onto embedded devices (investigate: because we use half-words and pack them could we get effective 1.5x bandwidth by doing a mix-in with words and half-words?)
    > [!INFO] 
    > we are likely memory (as-in, bandwidth) bound on an embedded device before being compute bound. However: (a) having this fancy hybrid architecture is awesome (b) why would we not utilize a stalled FPU? (I guess in low power mode) (c) it makes meeting real-time requirements easier if we can offload to the FPU or interleave with the FPU if we need more work done on CPU etc.

    - Levarge the CMSIS / VDSP / SIMD (AVX, SSE, NEON?) libs better
        - Via. Simde
        - Investigate possibility of StringZilla (https://github.com/ashvardanian/StringZilla)
        - Investigate possibility of SimSIMD https://github.com/ashvardanian/SimSIMD

    - Support dynamically rebuilding shader kernels

    - Build a merged PGO profile based on common tuning tables or some other heuristic / statistic,
    particularly for embedded targets

- Improving simulation quality of embedded hardware:
    - Empirically verify the results (or get close enough)
    - Heuristic for things like cache, pipeline stalling, or instruction cost Etc.
    - Account for unions, program memory, memory in transit etc. when tracking active memory use

- Improving embedded target consistency:
    - Ensure the daisy can get up and running properly and communicate all info over JTAG first

- Real time support on embedded targets:
    - A-lot ...

- Support other platforms / Arches:
    - RISCV support eventually planned for embedded + desktop
    - RTOS (FreeRTOS?) for embedded
    - Not opposed to supporting windows for desktop, but for now I'm ignoring it
