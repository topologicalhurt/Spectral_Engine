# Embedded-simulation end-to-end smoke (CTest).
#
# CI builds the simulate target on every push; this test also RUNS it: a real
# WAV through the full pipeline (analysis -> float->Q15 conversion -> the real
# spectral_arm32_process kernel -> WAV write), single-threaded — the only
# end-to-end host execution of the embedded path. Asserts a zero exit and a
# non-empty output WAV in an isolated working dir under the build tree.
#
# Run: cmake --build build --target simulate
#      && ctest --test-dir build -R simulate_smoke

# Nested one level below the build dir: the binary resolves its output dir
# relative to CWD and prefers "../output" when "../spectral_engine" exists,
# which is true at the build-dir top level (the build tree mirrors the source
# layout). One level down both probes miss and output lands inside WORK_DIR.
set(_smoke_work "${CMAKE_BINARY_DIR}/simulate_smoke_work/run")
file(MAKE_DIRECTORY "${_smoke_work}")

add_test(NAME simulate_smoke
    COMMAND ${CMAKE_COMMAND}
        "-DSIM_BIN=$<TARGET_FILE:simulate>"
        "-DINPUT_WAV=${SPECTRAL_REPO_ROOT}/resources/testing/sin_440hz.wav"
        "-DWORK_DIR=${_smoke_work}"
        -P "${CMAKE_CURRENT_LIST_DIR}/simulate-smoke-run.cmake")
