# SIMD width-parameterization parity harness (CTest).
#
# Q2 of QTYPE_DOMAIN_PLAN.md width-parameterizes the float L1 sustain kernel over
# SIMDE_NATURAL_FLOAT_VECTOR_SIZE (4-wide __m128 / 8-wide __m256). This test
# force-instantiates BOTH widths in one TU (SIMDe gives a portable __m256 on any
# host, including Apple Silicon NEON) and asserts 8-wide == 4-wide == scalar
# within the FMA-contraction budget. Complements osc_parity (which exercises the
# single production width vs scalar).
#
# Run: cmake --build build --target osc_width_parity_test \
#      && ctest --test-dir build -R osc_width_parity

add_executable(osc_width_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_osc_width_parity.c"
    "${SPECTRAL_CORE_DIR}/oscillator.c"
    "${SPECTRAL_CORE_DIR}/port/host/oscillator_simd.c"
    "${SPECTRAL_CORE_DIR}/spectral_osc_bandlimited.c"
    "${SPECTRAL_CORE_DIR}/spectral_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c"
    "${SPECTRAL_CORE_DIR}/spectral_log.c")

spectral_apply_common_target(osc_width_parity_test)
# The test #includes the width-templated kernel (.inc) and the scalar-wave header
# directly from core/port/host; that directory is not a global include path (the
# production .c reaches its siblings via the same-directory rule), so add it here.
target_include_directories(osc_width_parity_test PRIVATE "${SPECTRAL_CORE_DIR}/port/host")
target_link_libraries(osc_width_parity_test PRIVATE m)

if(APPLE)
    target_link_options(osc_width_parity_test PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(osc_width_parity_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(osc_width_parity_test PRIVATE -Wl,--gc-sections)
endif()

add_test(NAME osc_width_parity COMMAND osc_width_parity_test)
