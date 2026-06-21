# Log-power parabolic peak-interpolation contract (CTest).
#
# Exercises the real spectral_window_interp_magsq_parabolic (both its fast
# two-log branch and its overflow three-log fallback) against a double-precision
# reference. Replaces a tautological Python test that compared two identical
# Python helpers and never touched the C estimator.
#
# Run: cmake --build build --target peak_interp_parabolic_test
#      && ctest --test-dir build -R peak_interp_parabolic

add_executable(peak_interp_parabolic_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_peak_interp_parabolic.c"
    "${SPECTRAL_CORE_DIR}/spectral_windows.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c")

spectral_apply_common_target(peak_interp_parabolic_test)
target_compile_definitions(peak_interp_parabolic_test PRIVATE SPECTRAL_USE_VDSP=0)
target_link_libraries(peak_interp_parabolic_test PRIVATE m)

spectral_apply_dead_strip(peak_interp_parabolic_test)

add_test(NAME peak_interp_parabolic COMMAND peak_interp_parabolic_test)
