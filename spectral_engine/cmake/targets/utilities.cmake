# Utility targets: checks, bench, and build matrix/help display.

set(SPECTRAL_LOG_CHECK_FILES
    ${SPECTRAL_SOURCES_CORE}
    ${SPECTRAL_SOURCES_MONITORING}
    ${SPECTRAL_SOURCES_RUNTIME_PERF_MODEL}
    ${SPECTRAL_SOURCES_ANALYSIS}
    ${SPECTRAL_SOURCES_CLI}
    ${SPECTRAL_SOURCES_SYNTH_CPU}
    ${SPECTRAL_SOURCES_SYNTH_EMBEDDED}
    ${SPECTRAL_SOURCES_SYNTH_SIMULATION}
    ${SPECTRAL_SOURCES_SYNTH_CUDA}
    "${SPECTRAL_SOURCE_CONVERT_SEGMENTS_ENTRY}")
list(FILTER SPECTRAL_LOG_CHECK_FILES EXCLUDE REGEX "/core/spectral_log\\.c$")

find_program(RG_EXECUTABLE rg)
if(RG_EXECUTABLE)
    set(SPECTRAL_LOG_CHECK_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/spectral_log_check.cmake")
    file(WRITE "${SPECTRAL_LOG_CHECK_SCRIPT}" "set(LOG_CHECK_FILES\n")
    foreach(file_path IN LISTS SPECTRAL_LOG_CHECK_FILES)
        file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "  \"${file_path}\"\n")
    endforeach()
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" ")\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "message(STATUS \"Checking logging conformance in C/CUDA runtime and tool modules...\")\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "execute_process(COMMAND \"${RG_EXECUTABLE}\" -n \"\\\\b(printf|fprintf)\\\\s*\\\\(\" \${LOG_CHECK_FILES} RESULT_VARIABLE rg_result)\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "if(rg_result EQUAL 0)\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "  message(FATAL_ERROR \"Error: direct printf/fprintf found in modules covered by canonical logging policy. Use ${SPECTRAL_CORE_DIR}/spectral_log.h APIs.\")\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "elseif(NOT rg_result EQUAL 1)\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "  message(FATAL_ERROR \"rg failed while running logging conformance check (exit=\${rg_result}).\")\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "endif()\n")
    file(APPEND "${SPECTRAL_LOG_CHECK_SCRIPT}" "message(STATUS \"Logging conformance: OK\")\n")

    add_custom_target(log_check
        COMMAND ${CMAKE_COMMAND} -P "${SPECTRAL_LOG_CHECK_SCRIPT}")
else()
    add_custom_target(log_check
        COMMAND ${CMAKE_COMMAND} -E echo "rg not found; install ripgrep to run log checks.")
endif()

add_custom_target(syntax_test DEPENDS desktop embedded_arm embedded_arm_restricted)

separate_arguments(SPECTRAL_BENCH_ARGS_LIST NATIVE_COMMAND "${SPECTRAL_BENCH_ARGS}")
separate_arguments(SPECTRAL_BENCH_CACHE_ARGS_LIST NATIVE_COMMAND "${SPECTRAL_BENCH_CACHE_ARGS}")

add_custom_target(bench
    COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C bash "${SPECTRAL_BENCH_SCRIPT}"
            --binary "${SPECTRAL_BIN_DIR}/${TARGET_DESKTOP}"
            --input "${SPECTRAL_BENCH_INPUT}"
            --runs "${SPECTRAL_BENCH_RUNS}"
            --mode normal -- ${SPECTRAL_BENCH_ARGS_LIST}
    DEPENDS desktop
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}")

add_custom_target(bench_cache
    COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C bash "${SPECTRAL_BENCH_SCRIPT}"
            --binary "${SPECTRAL_BIN_DIR}/${TARGET_DESKTOP}"
            --input "${SPECTRAL_BENCH_INPUT}"
            --runs "${SPECTRAL_BENCH_RUNS}"
            --mode cache -- ${SPECTRAL_BENCH_CACHE_ARGS_LIST}
    DEPENDS desktop
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}")

add_custom_target(bench_all DEPENDS bench bench_cache)

add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "+--------------------------------------------------------------------------+"
    COMMAND ${CMAKE_COMMAND} -E echo "|                        SPECTRAL ENGINE BUILD MATRIX                      |"
    COMMAND ${CMAKE_COMMAND} -E echo "+--------------------------------------------------------------------------+"
    COMMAND ${CMAKE_COMMAND} -E echo "| Target                  | Analysis | Synth      | GPU   | Sample | Runs  |"
    COMMAND ${CMAKE_COMMAND} -E echo "+-------------------------+----------+------------+-------+--------+-------+"
    COMMAND ${CMAKE_COMMAND} -E echo "| desktop (default)       | YES      | CPU float  | YES   | float  | Desk  |"
    COMMAND ${CMAKE_COMMAND} -E echo "| simulate                | YES      | Q15 sim    | NO    | Q15    | Desk  |"
    COMMAND ${CMAKE_COMMAND} -E echo "| simulate_daisy          | NO       | Q15 sim    | NO    | Q15    | Desk  |"
    COMMAND ${CMAKE_COMMAND} -E echo "| embedded_arm            | YES      | Embedded   | NO    | Q15    | Host  |"
    COMMAND ${CMAKE_COMMAND} -E echo "| embedded_arm_float      | YES      | Embedded   | NO    | float  | Host  |"
    COMMAND ${CMAKE_COMMAND} -E echo "| embedded_arm_restricted | NO       | Embedded   | NO    | Q15    | Host  |"
    COMMAND ${CMAKE_COMMAND} -E echo "| daisy                   | NO       | (api/)     | NO    | Q15    | ARM   |"
    COMMAND ${CMAKE_COMMAND} -E echo "| daisy_example           | NO       | (examples) | NO    | Q15    | ARM   |"
    COMMAND ${CMAKE_COMMAND} -E echo "+-------------------------+----------+------------+-------+--------+-------+"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}"
    COMMAND ${CMAKE_COMMAND} -E echo "PGO mode: ${SPECTRAL_PGO}"
    COMMAND ${CMAKE_COMMAND} -E echo "Simulation board: ${SPECTRAL_SIMULATION_BOARD}"
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA enabled: ${SPECTRAL_USE_CUDA}"
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA target behavior: cuda -> desktop alias (no standalone CUDA demo binary)"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Desktop binary: ${SPECTRAL_BIN_DIR}/${TARGET_DESKTOP}"
    VERBATIM)

add_custom_target(spectral_help
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Spectral Engine CMake Build"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Configure (repo root): cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
    COMMAND ${CMAKE_COMMAND} -E echo "Build target: cmake --build build --target <target>"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Primary targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  desktop"
    COMMAND ${CMAKE_COMMAND} -E echo "  simulate"
    COMMAND ${CMAKE_COMMAND} -E echo "  simulate_daisy"
    COMMAND ${CMAKE_COMMAND} -E echo "  embedded_arm"
    COMMAND ${CMAKE_COMMAND} -E echo "  embedded_arm_float"
    COMMAND ${CMAKE_COMMAND} -E echo "  embedded_arm_restricted"
    COMMAND ${CMAKE_COMMAND} -E echo "  cuda (desktop alias; no standalone CUDA demo binary)"
    COMMAND ${CMAKE_COMMAND} -E echo "  daisy | daisy_flash | daisy_clean"
    COMMAND ${CMAKE_COMMAND} -E echo "  daisy_example | daisy_example_flash"
    COMMAND ${CMAKE_COMMAND} -E echo "  convert_segments"
    COMMAND ${CMAKE_COMMAND} -E echo "  log_check | syntax_test"
    COMMAND ${CMAKE_COMMAND} -E echo "  bench | bench_cache | bench_all"
    COMMAND ${CMAKE_COMMAND} -E echo "  info | spectral_help"
    VERBATIM)
