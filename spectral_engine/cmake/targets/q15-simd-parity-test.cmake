# packed 8xQ15 SIMD vs scalar Q15 parity harness (CTest, Q5c).
#
# Renders representative segments through the REAL production dispatch
# (timbre_synth_segment) with the opt-in Q15 compute domain on, toggling only the
# float scalar/SIMD axis (OSC_DISPATCH_ALL_SCALAR vs ALL_SIMD), and asserts the
# packed 8xQ15 SIMD kernel (osc_simd_q15_segment) matches its scalar Q15 oracle
# (synth_segment_q15) to the Q15-eval LSB floor. CI lock on the Q5c kernel: a lane-
# op / widen / amp-ramp regression fails the build. Same engine link set as
# q15_production_parity.
#
# Run: cmake --build build --target q15_simd_parity_test \
#      && ctest --test-dir build -R q15_simd_parity

add_executable(q15_simd_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_q15_simd_parity.c"
    "${SPECTRAL_CORE_DIR}/oscillator.c"
    "${SPECTRAL_ARCH_SIMD_DIR}/oscillator_simd.c"
    "${SPECTRAL_CORE_DIR}/spectral_osc_bandlimited.c"
    "${SPECTRAL_CORE_DIR}/spectral_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c"
    "${SPECTRAL_CORE_DIR}/spectral_log.c")

spectral_apply_common_target(q15_simd_parity_test)
# Expose the static-inline packed Q15 eval so the exhaustive parity sweep can call it.
target_compile_definitions(q15_simd_parity_test PRIVATE SPECTRAL_EXPOSE_Q15_PACK8_FOR_TEST=1)
target_link_libraries(q15_simd_parity_test PRIVATE m)

if(APPLE)
    target_link_options(q15_simd_parity_test PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(q15_simd_parity_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(q15_simd_parity_test PRIVATE -Wl,--gc-sections)
endif()

add_test(NAME q15_simd_parity COMMAND q15_simd_parity_test)
