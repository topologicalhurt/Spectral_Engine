# TODO's:

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

Ongoing:
- Improving throughput:
    - Double buffer CPU & GPU
    - Fix memory thrashing / page faulting for very low hop size in particular (or more generally, very large segment count)
    - Ensure segment binaries or files in VRAM while threads spinning up
    - Introducing FPU / float support onto embedded devices (investigate: because we use half-words and pack them could we get effective 1.5x bandwidth by doing a mix-in with
    words and half-words?)
    - Levarge the CMSIS lib better

- Improving emulation quality of embedded hardware:
    - Empirically verify the results (or get close enough)
    - Heuristic for things like cache, pipeline stalling, or instruction cost Etc.
    - Account for unions when tracking active memory use

- Improving embedded target consistency:
    - Ensure the daisy can get up and running properly and communicate all info over JTAG

- Real time support on embedded targets:
    - A-lot ...
