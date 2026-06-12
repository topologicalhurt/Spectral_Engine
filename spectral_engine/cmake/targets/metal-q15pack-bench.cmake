# GPU Q15 double-pack measure-first probe (manual benchmark, Apple-only).
#
# NOT a ctest: timing is machine- and GPU-dependent. Reproduction harness for
# docs/core_audit/archive/GPU_Q15_DOUBLEPACK_AUDIT.md — JIT-compiles MSL at
# runtime (as production spectral_synth_metal.m does) and times pure-sin and
# faithful-synth kernels in fp32 vs half/half2. Self-contained TU: no engine
# sources, only Metal/Foundation.
#
# Run: cmake --build build --target bench_metal_q15pack
#      && build/bin/bench_metal_q15pack

if(APPLE)
    add_executable(bench_metal_q15pack EXCLUDE_FROM_ALL
        "${SPECTRAL_REPO_ROOT}/tests/core_contracts/bench_metal_q15pack.m")
    spectral_apply_common_target(bench_metal_q15pack)
    target_compile_options(bench_metal_q15pack PRIVATE -fobjc-arc)
    target_link_libraries(bench_metal_q15pack PRIVATE
        "-framework Metal" "-framework Foundation")
endif()
