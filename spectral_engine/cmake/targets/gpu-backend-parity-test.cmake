# GPU (Metal) vs CPU backend parity (CTest, Apple-only).
#
# The only automated coverage of the Metal synthesis path: renders the same
# input through both backends via the real CLI and bounds the output delta
# (measured floors and bounds documented in the test script). Skips visibly
# (exit 77) when the desktop binary is not built or no Metal device exists
# (e.g. virtualized CI); a CPU fallback is never a silent pass.
#
# Run: cmake --build build --target desktop
#      && ctest --test-dir build -R gpu_backend_parity

if(APPLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME gpu_backend_parity
        COMMAND ${Python3_EXECUTABLE}
            "${SPECTRAL_REPO_ROOT}/tests/core_contracts/gpu_backend_parity.py"
            "$<TARGET_FILE:desktop>"
            "${SPECTRAL_REPO_ROOT}/resources/testing/sin_440hz.wav")
    set_tests_properties(gpu_backend_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()
