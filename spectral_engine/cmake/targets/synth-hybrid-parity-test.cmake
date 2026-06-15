# Hybrid IFFT/oscillator fast-path contract + parity (CTest, F-stream F2b).
#
# Compiles the hybrid router plus the frame renderer and BOTH host iFFT port TUs
# (each selects itself on SPECTRAL_USE_VDSP, so exactly one is live) and asserts:
# eligibility classification, deep-interior parity vs the exact oscillator sum at
# the F1 floor, and the decline-leaves-out-untouched contract.
#
# Run: cmake --build build --target synth_hybrid_parity_test
#      && ctest --test-dir build -R synth_hybrid_parity

add_executable(synth_hybrid_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_synth_hybrid_parity.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_hybrid.c"
    "${SPECTRAL_CORE_DIR}/spectral_synth_ifft.c"
    "${SPECTRAL_DRIVERS_VDSP_DIR}/spectral_ifft_vdsp.c"
    "${SPECTRAL_ARCH_REF_DIR}/spectral_ifft_ref.c")

spectral_apply_common_target(synth_hybrid_parity_test)
target_link_libraries(synth_hybrid_parity_test PRIVATE m)
if(APPLE)
    target_link_libraries(synth_hybrid_parity_test PRIVATE "-framework Accelerate")
endif()

add_test(NAME synth_hybrid_parity COMMAND synth_hybrid_parity_test)
