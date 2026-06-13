# Q5c packed-8xQ15 measure-first probe (manual benchmark).
#
# NOT a ctest: timing is machine-dependent, so this is a standalone EXCLUDE_FROM_ALL
# executable, run by hand, numbers reported in patch notes. It renders the same
# sustain segment three ways -- production 4-wide float-SIMD, packed 8xQ15 with float
# accumulate (desktop), packed 8xQ15 with Q15 accumulate (embedded ceiling) -- to
# settle whether a production width-templated Q15 kernel earns its keep on desktop.
# Same engine link set as bench_q15_throughput.
#
# Run: cmake --build build --target bench_q15_pack8 && build/bench_q15_pack8

add_executable(bench_q15_pack8 EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/bench_q15_pack8.c"
    "${SPECTRAL_CORE_DIR}/spectral_oscillator.c"
    "${SPECTRAL_ARCH_SIMD_DIR}/oscillator_simd.c"
    "${SPECTRAL_CORE_DIR}/spectral_osc_bandlimited.c"
    "${SPECTRAL_CORE_DIR}/spectral_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c")

spectral_apply_common_target(bench_q15_pack8)
target_link_libraries(bench_q15_pack8 PRIVATE m)

if(APPLE)
    target_link_options(bench_q15_pack8 PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(bench_q15_pack8 PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(bench_q15_pack8 PRIVATE -Wl,--gc-sections)
endif()
