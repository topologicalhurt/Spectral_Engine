# Utility targets: checks, bench, and build matrix/help display.

include("${SPECTRAL_ENGINE_ROOT}/cmake/python_env.cmake")

set(SPECTRAL_RESOURCE_HASH_SCRIPT "${SPECTRAL_REPO_ROOT}/tools/spectral_tools/generators/resource_hashes.py")
set(SPECTRAL_RESOURCE_HASH_OUTPUT "${SPECTRAL_CORE_DIR}/spectral_hash_resources_xx32_xx3.c")
set(SPECTRAL_RESOURCE_HASH_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/run_resource_hashes.cmake")
configure_file(
    "${SPECTRAL_ENGINE_ROOT}/cmake/scripts/run_resource_hashes.cmake.in"
    "${SPECTRAL_RESOURCE_HASH_RUNNER}"
    @ONLY)
file(GLOB_RECURSE SPECTRAL_FIRMWARE_RESOURCE_FILES CONFIGURE_DEPENDS
    "${SPECTRAL_REPO_ROOT}/resources/*")

add_custom_command(
    OUTPUT "${SPECTRAL_RESOURCE_HASH_OUTPUT}"
    COMMAND ${CMAKE_COMMAND}
            -DSPECTRAL_HASH_MODE=generate
            -P "${SPECTRAL_RESOURCE_HASH_RUNNER}"
    DEPENDS
        "${SPECTRAL_PYTHON_ENV_STAMP}"
        "${SPECTRAL_RESOURCE_HASH_SCRIPT}"
        "${SPECTRAL_RESOURCE_HASH_RUNNER}"
        ${SPECTRAL_FIRMWARE_RESOURCE_FILES}
    VERBATIM)

add_custom_target(generate_resource_hashes
    DEPENDS "${SPECTRAL_RESOURCE_HASH_OUTPUT}")
add_dependencies(generate_resource_hashes prepare_python_tools spectral_resource_bridge)

set_source_files_properties(
    "${SPECTRAL_RESOURCE_HASH_OUTPUT}"
    PROPERTIES GENERATED TRUE)

add_custom_target(verify_resource_hashes
    COMMAND ${CMAKE_COMMAND}
            -DSPECTRAL_HASH_MODE=verify
            -P "${SPECTRAL_RESOURCE_HASH_RUNNER}"
    DEPENDS
        prepare_python_tools
        spectral_resource_bridge
        generate_resource_hashes
        "${SPECTRAL_RESOURCE_HASH_SCRIPT}"
        "${SPECTRAL_RESOURCE_HASH_RUNNER}"
        ${SPECTRAL_FIRMWARE_RESOURCE_FILES}
    VERBATIM)

if(TARGET desktop)
    target_sources(desktop PRIVATE "${SPECTRAL_RESOURCE_HASH_OUTPUT}")
    add_dependencies(desktop verify_resource_hashes)
endif()
if(TARGET simulate)
    target_sources(simulate PRIVATE "${SPECTRAL_RESOURCE_HASH_OUTPUT}")
    add_dependencies(simulate verify_resource_hashes)
endif()
if(TARGET simulate_daisy)
    target_sources(simulate_daisy PRIVATE "${SPECTRAL_RESOURCE_HASH_OUTPUT}")
    add_dependencies(simulate_daisy verify_resource_hashes)
endif()
if(TARGET embedded_arm)
    target_sources(embedded_arm PRIVATE "${SPECTRAL_RESOURCE_HASH_OUTPUT}")
    add_dependencies(embedded_arm verify_resource_hashes)
endif()
if(TARGET embedded_arm_float)
    target_sources(embedded_arm_float PRIVATE "${SPECTRAL_RESOURCE_HASH_OUTPUT}")
    add_dependencies(embedded_arm_float verify_resource_hashes)
endif()
if(TARGET embedded_arm_restricted)
    target_sources(embedded_arm_restricted PRIVATE "${SPECTRAL_RESOURCE_HASH_OUTPUT}")
    add_dependencies(embedded_arm_restricted verify_resource_hashes)
endif()

set(SPECTRAL_LOG_CHECK_FILES
    ${SPECTRAL_SOURCES_CORE}
    ${SPECTRAL_SOURCES_CORE_OSC_SIMD_HOST}
    ${SPECTRAL_SOURCES_CORE_OSC_SIMD_EMBEDDED}
    ${SPECTRAL_SOURCES_CORE_VECTOR_OPS_HOST}
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

add_custom_target(bench
    COMMAND ${CMAKE_COMMAND} -E env
            LC_ALL=C
            "PYTHONPATH=${SPECTRAL_TOOLS_PYTHONPATH}"
            "${SPECTRAL_PYTHON}" -m "${SPECTRAL_BENCH_MODULE}"
            bench
            --binary "${SPECTRAL_BIN_DIR}/${TARGET_DESKTOP}"
            --input "${SPECTRAL_BENCH_INPUT}"
            --runs "${SPECTRAL_BENCH_RUNS}"
            --mode normal
            --bench-args "${SPECTRAL_BENCH_ARGS}"
    DEPENDS prepare_python_tools desktop
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}")

add_custom_target(bench_cache
    COMMAND ${CMAKE_COMMAND} -E env
            LC_ALL=C
            "PYTHONPATH=${SPECTRAL_TOOLS_PYTHONPATH}"
            "${SPECTRAL_PYTHON}" -m "${SPECTRAL_BENCH_MODULE}"
            bench
            --binary "${SPECTRAL_BIN_DIR}/${TARGET_DESKTOP}"
            --input "${SPECTRAL_BENCH_INPUT}"
            --runs "${SPECTRAL_BENCH_RUNS}"
            --mode cache
            --bench-args "${SPECTRAL_BENCH_CACHE_ARGS}"
    DEPENDS prepare_python_tools desktop
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}")

add_custom_target(bench_all DEPENDS bench bench_cache)
add_custom_target(pgo_collect DEPENDS bench bench_cache)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    find_program(SPECTRAL_LLVM_PROFDATA_EXECUTABLE llvm-profdata)
    if(SPECTRAL_LLVM_PROFDATA_EXECUTABLE)
        set(SPECTRAL_PGO_MERGE_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/spectral_pgo_merge.cmake")
        file(WRITE "${SPECTRAL_PGO_MERGE_SCRIPT}"
"file(MAKE_DIRECTORY \"${SPECTRAL_PGO_DIR}\")
file(GLOB _spectral_profraw_files \"${SPECTRAL_PGO_DIR}/*.profraw\")
if(NOT _spectral_profraw_files)
    message(FATAL_ERROR \"No .profraw files found in ${SPECTRAL_PGO_DIR}. Run pgo_collect with SPECTRAL_PGO=gen first.\")
endif()
execute_process(
    COMMAND \"${SPECTRAL_LLVM_PROFDATA_EXECUTABLE}\" merge -output=\"${SPECTRAL_PGO_PROFILE}\" \${_spectral_profraw_files}
    RESULT_VARIABLE _spectral_merge_result)
if(NOT _spectral_merge_result EQUAL 0)
    message(FATAL_ERROR \"llvm-profdata merge failed (exit=\${_spectral_merge_result})\")
endif()
message(STATUS \"Wrote merged profile: ${SPECTRAL_PGO_PROFILE}\")
")
        add_custom_target(pgo_merge
            COMMAND ${CMAKE_COMMAND} -P "${SPECTRAL_PGO_MERGE_SCRIPT}"
            VERBATIM)
    else()
        add_custom_target(pgo_merge
            COMMAND ${CMAKE_COMMAND} -E echo "llvm-profdata not found; install LLVM tools to merge Clang .profraw files.")
    endif()
else()
    add_custom_target(pgo_merge
        COMMAND ${CMAKE_COMMAND} -E echo "GCC PGO does not require profraw merge. Use SPECTRAL_PGO=use after profile run.")
endif()

add_custom_target(pgo_help
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "PGO workflow (recommended release-fast build):"
    COMMAND ${CMAKE_COMMAND} -E echo "  1) Configure gen: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPECTRAL_PRODUCTION_BUILD=OFF -DSPECTRAL_PGO=gen"
    COMMAND ${CMAKE_COMMAND} -E echo "  2) Build + run training: cmake --build build --target pgo_collect"
    COMMAND ${CMAKE_COMMAND} -E echo "  3) Clang only: cmake --build build --target pgo_merge"
    COMMAND ${CMAKE_COMMAND} -E echo "  4) Reconfigure use: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPECTRAL_PRODUCTION_BUILD=OFF -DSPECTRAL_PGO=use"
    COMMAND ${CMAKE_COMMAND} -E echo "  5) Rebuild optimized: cmake --build build --target desktop"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "PGO directory: ${SPECTRAL_PGO_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "Clang raw pattern: ${SPECTRAL_PGO_RAW_PATTERN}"
    COMMAND ${CMAKE_COMMAND} -E echo "Clang merged profile: ${SPECTRAL_PGO_PROFILE}"
    VERBATIM)

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
    COMMAND ${CMAKE_COMMAND} -E echo "Configure (repo root): cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug"
    COMMAND ${CMAKE_COMMAND} -E echo "Production configure: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPECTRAL_PRODUCTION_BUILD=ON"
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
    COMMAND ${CMAKE_COMMAND} -E echo "  pgo_collect | pgo_merge | pgo_help"
    COMMAND ${CMAKE_COMMAND} -E echo "  info | spectral_help"
    VERBATIM)
