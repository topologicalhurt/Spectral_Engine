# Coupled-form Q31 recursive oscillator: fixed-point SNR/stability contract (CTest, C1).
#
# Make-or-break gate for the LUT-gather replacement (TEMP_MEM_OPT_PLAN C1). Steps the
# coupled-form oscillator (spectral_osc_recursive.h) for 8 s per frequency across a sweep,
# with per-block renorm, and asserts SNR >= 72 dB vs exact sin plus a drift bound -- the
# precision/stability the magic-circle form failed. Header-only + the test TU; nothing in
# the engine links it yet.
#
# Run: cmake --build build --target osc_recursive_test \
#      && ctest --test-dir build -R osc_recursive

add_executable(osc_recursive_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_math/test_osc_recursive.c")

spectral_apply_common_target(osc_recursive_test)
target_link_libraries(osc_recursive_test PRIVATE m)

add_test(NAME osc_recursive COMMAND osc_recursive_test)
