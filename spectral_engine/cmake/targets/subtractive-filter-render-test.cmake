# Subtractive source x filter integration (CTest) — RENDERER_ABSTRACTION_PLAN.md, Stage 3b-2.
#
# End-to-end: saw source -> harmonics (3a-1) -> expand -> low-pass filter (3b-1) -> IFFT render,
# then independently measure the output spectrum and assert it equals source x H(f). Links the
# harmonics bridge + the filter envelope + the IFFT frame renderer + both host iFFT port TUs
# (mirrors wavetable-render-parity-test).
#
# Run: cmake --build build --target subtractive_filter_render_test \
#      && ctest --test-dir build -R subtractive_filter_render

add_executable(subtractive_filter_render_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_subtractive_filter_render.c"
    "${SPECTRAL_CORE_DIR}/spectral_wavetable_harmonics.c"
    "${SPECTRAL_CORE_DIR}/spectral_filter_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_ifft.c"
    "${SPECTRAL_DRIVERS_VDSP_DIR}/spectral_ifft_vdsp.c"
    "${SPECTRAL_ARCH_REF_DIR}/spectral_ifft_ref.c")

spectral_apply_common_target(subtractive_filter_render_test)
target_link_libraries(subtractive_filter_render_test PRIVATE m)
if(APPLE)
    target_link_libraries(subtractive_filter_render_test PRIVATE "-framework Accelerate")
endif()
spectral_apply_dead_strip(subtractive_filter_render_test)

add_test(NAME subtractive_filter_render COMMAND subtractive_filter_render_test)
