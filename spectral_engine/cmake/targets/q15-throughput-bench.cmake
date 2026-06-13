# opt-in Q15 throughput probe (manual benchmark, Q3b #75).
#
# NOT a ctest: timing is machine-dependent, so this is a standalone EXCLUDE_FROM_ALL
# executable, run by hand, numbers reported in patch notes. It renders the same
# segment through production timbre_synth_segment() under float-scalar / q15-scalar
# / float-simd to settle whether the opt-in Q15 path earns a SIMD kernel. Same
# engine link set as q15_production_parity_test.
#
# Run: cmake --build build --target bench_q15_throughput \
#      && build/bench_q15_throughput

add_executable(bench_q15_throughput EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/bench_q15_throughput.c"
    "${SPECTRAL_CORE_DIR}/spectral_oscillator.c"
    "${SPECTRAL_ARCH_SIMD_DIR}/oscillator_simd.c"
    "${SPECTRAL_CORE_DIR}/spectral_osc_bandlimited.c"
    "${SPECTRAL_CORE_DIR}/spectral_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c")

spectral_apply_common_target(bench_q15_throughput)
target_link_libraries(bench_q15_throughput PRIVATE m)

if(APPLE)
    target_link_options(bench_q15_throughput PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(bench_q15_throughput PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(bench_q15_throughput PRIVATE -Wl,--gc-sections)
endif()
