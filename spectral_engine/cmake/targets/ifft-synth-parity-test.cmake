# IFFT synthesis contract + parity (CTest, F-stream F2).
#
# Compiles the frame renderer plus BOTH host iFFT port TUs (each selects
# itself on SPECTRAL_USE_VDSP, so exactly one is live) and asserts the
# parity ladder: port-vs-reference-iDFT, stream parity vs the exact
# oscillator sum at the F1-measured floors, determinism.
#
# Run: cmake --build build --target ifft_synth_parity_test
#      && ctest --test-dir build -R ifft_synth_parity

add_executable(ifft_synth_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_ifft_synth_parity.c"
    "${SPECTRAL_ENGINE_ROOT}/synth/spectral_synth_ifft.c"
    "${SPECTRAL_CORE_PORT_HOST_DIR}/spectral_ifft_vdsp.c"
    "${SPECTRAL_CORE_PORT_HOST_DIR}/spectral_ifft_ref.c")

spectral_apply_common_target(ifft_synth_parity_test)
target_include_directories(ifft_synth_parity_test PRIVATE
    "${SPECTRAL_ENGINE_ROOT}/synth")
target_link_libraries(ifft_synth_parity_test PRIVATE m)
if(APPLE)
    target_link_libraries(ifft_synth_parity_test PRIVATE "-framework Accelerate")
endif()

add_test(NAME ifft_synth_parity COMMAND ifft_synth_parity_test)
