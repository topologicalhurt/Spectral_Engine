# Double-precision init-sine parity contract (CTest).
#
# Gates spectral_osc_sin_init_f64 (spectral_osc_q31.h) -- the reference-grade
# double sine that seeds the coupled-form Q31 oscillator's rotation constants --
# against libm sin over its REAL caller domain [0, 2*pi + pi/2) (omega/phi from a
# uint32 phase * 2*pi/2^32, evaluated at angle and angle+pi/2 by
# spectral_coupled_init). Tolerance 1e-9 = the header's stated rotation-constant
# accuracy; measured residual is 6e-12, and a Taylor-degree drop or a float
# substitution (the regressions the header warns about) overshoot it ~3500x.
#
# Header-only + the test TU; nothing in the engine links it. Modeled on
# osc-recursive-test.cmake (same header, same dependency shape).
#
# Run: cmake --build build --target osc_sin_init_parity_test \
#      && ctest --test-dir build -R osc_sin_init_parity

add_executable(osc_sin_init_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_osc_sin_init_parity.c")

spectral_apply_common_target(osc_sin_init_parity_test)
target_link_libraries(osc_sin_init_parity_test PRIVATE m)

add_test(NAME osc_sin_init_parity COMMAND osc_sin_init_parity_test)
