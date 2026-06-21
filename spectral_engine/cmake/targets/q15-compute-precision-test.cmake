# Q15-compute precision characterization harness (CTest, Q3a).
#
# Renders the L1 oscillator waveforms in pure Q15 and against the float L0
# formulas (driven by the same phase), reporting per-timbre RMS error in dBFS.
# This is the measure-first input to QTYPE_DOMAIN_PLAN.md Q3's per-path "does this
# clear the 15-bit bar?" decision. It changes no production output — header-only
# Q15/LUT/formula inlines plus the test TU, nothing linked from the engine.
#
# Run: cmake --build build --target q15_compute_precision_test \
#      && ctest --test-dir build -R q15_compute_precision

add_executable(q15_compute_precision_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_q15_compute_precision.c")

spectral_apply_common_target(q15_compute_precision_test)
target_link_libraries(q15_compute_precision_test PRIVATE m)

add_test(NAME q15_compute_precision COMMAND q15_compute_precision_test)
