# opt-in Q15 vs float production parity harness (CTest, Q3b).
#
# Renders representative segments through the REAL production dispatch
# (timbre_synth_segment) with the opt-in Q15 compute domain off (float, the
# shipping default) and on, and asserts the per-timbre RMS error stays at the
# characterized Q15 floor (q15_compute_precision). This is the CI lock
# on the QTYPE_DOMAIN_PLAN.md §7 per-path sign-off: float stays default, a Q15
# regression fails the build. Same engine link set as osc_parity.
#
# Run: cmake --build build --target q15_production_parity_test \
#      && ctest --test-dir build -R q15_production_parity

add_executable(q15_production_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_q15_production_parity.c"
    "${SPECTRAL_CORE_DIR}/spectral_oscillator.c"
    "${SPECTRAL_ARCH_SIMD_DIR}/spectral_oscillator_simd.c"
    "${SPECTRAL_CORE_DIR}/spectral_osc_bandlimited.c"
    "${SPECTRAL_CORE_DIR}/spectral_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c"
    "${SPECTRAL_CORE_DIR}/spectral_log.c")

spectral_apply_common_target(q15_production_parity_test)
target_link_libraries(q15_production_parity_test PRIVATE m)

spectral_apply_dead_strip(q15_production_parity_test)

add_test(NAME q15_production_parity COMMAND q15_production_parity_test)
