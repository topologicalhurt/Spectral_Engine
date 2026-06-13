# SIMD-vs-scalar oscillator parity harness (CTest).
#
# Compiled (not string-grep) guard that the SIMD CPU oscillator stays within the
# documented per-sample drift budget and aggregate RMS-dBFS budget of the scalar
# reference, for every SIMD-capable timbre. Both paths run through the real
# production dispatch (timbre_synth_segment under osc_set_dispatch), so this pins
# the equivalence the SIMD-default oscillator baseline rests on.
#
# Run: cmake --build build --target osc_parity_test && ctest --test-dir build -R osc_parity

add_executable(osc_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_osc_parity.c"
    "${SPECTRAL_CORE_DIR}/spectral_oscillator.c"
    "${SPECTRAL_ARCH_SIMD_DIR}/spectral_oscillator_simd.c"
    "${SPECTRAL_CORE_DIR}/spectral_osc_bandlimited.c"
    "${SPECTRAL_CORE_DIR}/spectral_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c"
    "${SPECTRAL_CORE_DIR}/spectral_log.c")

spectral_apply_common_target(osc_parity_test)
target_link_libraries(osc_parity_test PRIVATE m)

if(APPLE)
    target_link_options(osc_parity_test PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(osc_parity_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(osc_parity_test PRIVATE -Wl,--gc-sections)
endif()

add_test(NAME osc_parity COMMAND osc_parity_test)
