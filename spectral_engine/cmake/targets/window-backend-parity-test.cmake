# Window backend-parity contract (CTest).
#
# Asserts spectral_window_hann/hamming/blackman match the documented symmetric
# (N-1) raised-cosine SSOT. Compiled with SPECTRAL_USE_VDSP=1 (desktop default)
# so a reintroduced vDSP periodic window would fail the symmetric reference.
#
# Run: cmake --build build --target window_backend_parity_test
#      && ctest --test-dir build -R window_backend_parity

add_executable(window_backend_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_window_backend_parity.c"
    "${SPECTRAL_CORE_DIR}/spectral_windows.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c")

spectral_apply_common_target(window_backend_parity_test)
# Force VDSP=1: the windows must be symmetric even when the Apple-accelerated
# path would otherwise be selected (it no longer exists, and this guards that).
target_compile_definitions(window_backend_parity_test PRIVATE SPECTRAL_USE_VDSP=1)
target_link_libraries(window_backend_parity_test PRIVATE m)

if(APPLE)
    target_link_options(window_backend_parity_test PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(window_backend_parity_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(window_backend_parity_test PRIVATE -Wl,--gc-sections)
endif()

add_test(NAME window_backend_parity COMMAND window_backend_parity_test)
