# Wavetable IFFT fast-path contract + parity (CTest) — RENDERER_ABSTRACTION_PLAN.md, Stage 3a-3.
#
# The wavetable analog of synth-hybrid-parity-test: compiles the wavetable hybrid + the harmonics
# bridge + the sine hybrid (for the shared segment-eligibility gate) + the frame renderer + both
# host iFFT port TUs, and asserts RENDERED on an eligible dense scene, deep-interior parity vs the
# band-limited additive sum, and the decline-leaves-out-untouched contract. Like the sine hybrid,
# none of this is in the engine library — it is opt-in / F3-gated.
#
# Run: cmake --build build --target synth_hybrid_wavetable_parity_test \
#      && ctest --test-dir build -R synth_hybrid_wavetable_parity

add_executable(synth_hybrid_wavetable_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_synth_hybrid_wavetable_parity.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_hybrid_wavetable.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_hybrid.c"
    "${SPECTRAL_CORE_DIR}/spectral_wavetable_harmonics.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_ifft.c"
    "${SPECTRAL_DRIVERS_VDSP_DIR}/spectral_ifft_vdsp.c"
    "${SPECTRAL_ARCH_REF_DIR}/spectral_ifft_ref.c")

spectral_apply_common_target(synth_hybrid_wavetable_parity_test)
target_link_libraries(synth_hybrid_wavetable_parity_test PRIVATE m)
if(APPLE)
    target_link_libraries(synth_hybrid_wavetable_parity_test PRIVATE "-framework Accelerate")
endif()
spectral_apply_dead_strip(synth_hybrid_wavetable_parity_test)

add_test(NAME synth_hybrid_wavetable_parity COMMAND synth_hybrid_wavetable_parity_test)
