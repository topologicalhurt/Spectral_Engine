# Band-limited (additive) oscillator vs analytic Fourier-series contract (CTest).
#
# Pins spectral_osc_bandlimited_synth_segment(ADDITIVE) for saw/square/triangle/parabola/sine
# against an independent double-precision libm sum of the same Nyquist-capped series. Catches
# a wrong harmonic coefficient, a sin/cos quadrature swap, an off-by-one in n_harm, or a
# regression in the per-timbre sin/cos recurrence specialization. The module is otherwise
# uncovered.
#
# Run: cmake --build build --target osc_bandlimited_parity_test
#      && ctest --test-dir build -R osc_bandlimited_parity

add_executable(osc_bandlimited_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_osc_bandlimited_parity.c"
    "${SPECTRAL_CORE_DIR}/spectral_osc_bandlimited.c"
    "${SPECTRAL_CORE_DIR}/spectral_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c")

spectral_apply_common_target(osc_bandlimited_parity_test)
target_link_libraries(osc_bandlimited_parity_test PRIVATE m)

add_test(NAME osc_bandlimited_parity COMMAND osc_bandlimited_parity_test)
