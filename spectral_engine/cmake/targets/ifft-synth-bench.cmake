# IFFT-synthesis vs oscillator-loop speedup probe (manual benchmark, F-stream F2).
#
# NOT a ctest: timing is machine-dependent, so this is a standalone EXCLUDE_FROM_ALL
# executable, run by hand, numbers reported in patch notes. It sweeps partial count
# and measures the IFFT path vs the naive oscillator loop to verify the ~8x at high
# density and locate the crossover. Same iFFT port link set as the parity test.
#
# Run: cmake --build build --target bench_ifft_synth && build/bench_ifft_synth

add_executable(bench_ifft_synth EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/bench_ifft_synth.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_ifft.c"
    "${SPECTRAL_DRIVERS_VDSP_DIR}/spectral_ifft_vdsp.c"
    "${SPECTRAL_ARCH_REF_DIR}/spectral_ifft_ref.c")

spectral_apply_common_target(bench_ifft_synth)
target_link_libraries(bench_ifft_synth PRIVATE m)
if(APPLE)
    target_link_libraries(bench_ifft_synth PRIVATE "-framework Accelerate")
endif()
