# Wavetable frequency-domain render parity (CTest) — RENDERER_ABSTRACTION_PLAN.md, Stage 3a-2.
#
# End-to-end: extract a table's harmonics, expand wavetable partials into additive partials, and
# render them through the real IFFT synth; assert it matches the exact band-limited additive sum
# at the IFFT floor. Links the harmonics bridge + the frame renderer + BOTH host iFFT port TUs
# (each selects itself on SPECTRAL_USE_VDSP, so exactly one is live), mirroring
# ifft-synth-parity-test.cmake.
#
# Run: cmake --build build --target wavetable_render_parity_test \
#      && ctest --test-dir build -R wavetable_render_parity

add_executable(wavetable_render_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_wavetable_render_parity.c"
    "${SPECTRAL_CORE_DIR}/spectral_wavetable_harmonics.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_ifft.c"
    "${SPECTRAL_DRIVERS_VDSP_DIR}/spectral_ifft_vdsp.c"
    "${SPECTRAL_ARCH_REF_DIR}/spectral_ifft_ref.c")

spectral_apply_common_target(wavetable_render_parity_test)
target_link_libraries(wavetable_render_parity_test PRIVATE m)
if(APPLE)
    target_link_libraries(wavetable_render_parity_test PRIVATE "-framework Accelerate")
endif()
spectral_apply_dead_strip(wavetable_render_parity_test)

add_test(NAME wavetable_render_parity COMMAND wavetable_render_parity_test)
