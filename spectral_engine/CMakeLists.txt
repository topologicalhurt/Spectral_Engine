cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR
        "Do not configure from spectral_engine/ directly.\n"
        "Configure from repository root instead:\n"
        "  cmake -S . -B build")
endif()

project(SpectralEngine LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Default to Debug for single-config generators unless explicitly set.
set(SPECTRAL_BUILD_TYPE_DEFAULTED OFF)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING
        "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    set(SPECTRAL_BUILD_TYPE_DEFAULTED ON)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)

set(SPECTRAL_ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(SPECTRAL_REPO_ROOT "${SPECTRAL_ENGINE_ROOT}/.." ABSOLUTE)
include("${SPECTRAL_ENGINE_ROOT}/cmake/options.cmake")

if(SPECTRAL_DEBUG_LOGGING)
    add_compile_definitions(SPECTRAL_DEBUG=1)
endif()
if(SPECTRAL_REPRO_BUILD)
    add_compile_definitions(SPECTRAL_CUSTOM_FAST_MATH_MODE=0)
    set(SPECTRAL_CUSTOM_FAST_MATH_MODE_STATUS OFF)
else()
    add_compile_definitions(SPECTRAL_CUSTOM_FAST_MATH_MODE=1)
    set(SPECTRAL_CUSTOM_FAST_MATH_MODE_STATUS ON)
endif()
add_compile_definitions(SPECTRAL_TRACK_PAIR_OMP_CHUNK=${SPECTRAL_TRACK_PAIR_OMP_CHUNK_CANONICAL})

if(SPECTRAL_USE_CUDA)
    # CUDA 12.4 supports GCC <= 13.  Pin nvcc's host compiler to g++-13
    # while the rest of the build uses the system GCC (14).  The desktop
    # target forces LINKER_LANGUAGE C so the final link is driven by GCC 14
    # (which can read its own LTO bytecode).
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        find_program(_cuda_host_13 g++-13)
        if(_cuda_host_13)
            set(CMAKE_CUDA_HOST_COMPILER "${_cuda_host_13}")
        endif()
        unset(_cuda_host_13)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(APPLE)
    enable_language(OBJC)
endif()

if(NOT SPECTRAL_SKIP_VERSION_CHECK)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        if(CMAKE_C_COMPILER_VERSION VERSION_LESS "14.0.0")
            message(FATAL_ERROR "${CMAKE_C_COMPILER} version ${CMAKE_C_COMPILER_VERSION} is too old. Minimum: 14.0.0")
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_C_COMPILER_VERSION VERSION_LESS "11.0.0")
            message(FATAL_ERROR "${CMAKE_C_COMPILER} version ${CMAKE_C_COMPILER_VERSION} is too old. Minimum: 11.0.0")
        endif()
    endif()
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SPECTRAL_ARCH)
if(SPECTRAL_ARCH STREQUAL "")
    set(SPECTRAL_ARCH unknown)
endif()

if(APPLE)
    set(SPECTRAL_GPU_BACKEND metal)
elseif(SPECTRAL_USE_CUDA)
    set(SPECTRAL_GPU_BACKEND cuda)
else()
    set(SPECTRAL_GPU_BACKEND native)
endif()

set(TARGET_DESKTOP "spectral_${SPECTRAL_ARCH}_${SPECTRAL_GPU_BACKEND}_desktop")
set(TARGET_SIMULATION "spectral_${SPECTRAL_ARCH}_native_simulation")
set(TARGET_SIMULATION_DAISY "spectral_${SPECTRAL_ARCH}_native_simulation_daisy")
set(TARGET_EMBEDDED_ARM "spectral_embedded_arm32")
set(TARGET_EMBEDDED_ARM_FLOAT "spectral_embedded_arm32_float")
set(TARGET_EMBEDDED_ARM_RES "spectral_embedded_arm32_restricted")

set(SPECTRAL_BIN_DIR "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SPECTRAL_BIN_DIR}")
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${SPECTRAL_BIN_DIR}")
endforeach()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/source-manifest.cmake")

set(SPECTRAL_XXHASH_SOURCE "${SPECTRAL_REPO_ROOT}/third_party/xxHash/xxhash.c")
add_library(spectral_xxhash STATIC "${SPECTRAL_XXHASH_SOURCE}")
target_include_directories(spectral_xxhash PUBLIC "${SPECTRAL_XXHASH_INCLUDE_DIR}")

# Python ctypes bridge — standalone .so for resource_hashes.py
add_library(spectral_resource_bridge SHARED
    core/spectral_resource_fs.c
    core/spectral_resource_bridge.c
)
target_include_directories(spectral_resource_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
)
set_target_properties(spectral_resource_bridge PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

set(SPECTRAL_INCLUDE_DIRS ${SPECTRAL_ENGINE_INCLUDE_DIRS})

set(SPECTRAL_SYSTEM_INCLUDE_DIRS
    "${SPECTRAL_SIMDE_INCLUDE_DIR}"
    "${SPECTRAL_XXHASH_INCLUDE_DIR}")

set(SPECTRAL_SIMULATION_DEFINES
    SPECTRAL_EMBEDDED=1
    SPECTRAL_USE_EMBEDDED_SYNTH=1
    SPECTRAL_EMBEDDED_SIMULATION=1)

set(SPECTRAL_SIMULATION_DAISY_DEFINES
    ${SPECTRAL_SIMULATION_DEFINES}
    SPECTRAL_RESTRICTED_MODE=1)

# Extra defines layered on top of SPECTRAL_SIMULATION_DEFINES for the embedded
# host variants (see cmake/targets/embedded.cmake). Without these the variables
# expand to empty and embedded_arm_float silently equals embedded_arm while
# embedded_arm_restricted fails to link (its CLI pipeline pulls in analyze_audio
# and perf_*, which RESTRICTED_MODE/NO_PERF guard out).
set(SPECTRAL_EMBEDDED_FLOAT_DEFINES
    SPECTRAL_EMBEDDED_FLOAT=1)

set(SPECTRAL_EMBEDDED_RESTRICTED_DEFINES
    SPECTRAL_RESTRICTED_MODE=1
    SPECTRAL_NO_PERF=1)


include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/host-config.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/desktop.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/simulation.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/embedded.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/cuda.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/convert-segments.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/daisy.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/utilities.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/metal-osc-codegen.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/arm-core-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/core-contracts-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/core-guarantees-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/osc-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/osc-width-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q-domain-contract-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/osc-backend-contract-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q-ladder-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-compute-precision-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-production-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-simd-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-throughput-bench.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-pack8-bench.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/metal-q15pack-bench.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/phase-nco-precision-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/osc-recursive-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/osc-sin-init-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/proc-mask-honesty-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/full-fused-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/osc-formulas-domain-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/window-backend-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/peak-interp-parabolic-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/segment-endian-roundtrip-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/segment-payload-bounds-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/ifft-synth-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/simulate-smoke-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/gpu-backend-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/daisy-config-layout-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/convert-segments-units-test.cmake")

# Test/bench targets that compile core/spectral_oscillator.c also #include the generated
# spectral_osc_metal_generated.h. They are defined AFTER metal-osc-codegen.cmake,
# so its own foreach (over the main build targets) cannot reach them — without
# this, a parallel `tests_all` can compile spectral_oscillator.c while the codegen custom
# command is mid-regeneration of the header into the source tree (a known flake).
# Gate each on generate_metal_osc so the header is present before they compile.
foreach(_spectral_osc_test_target
        osc_parity_test osc_width_parity_test
        q15_production_parity_test q15_simd_parity_test
        full_fused_parity_test
        bench_q15_pack8 bench_q15_throughput)
    if(TARGET ${_spectral_osc_test_target})
        add_dependencies(${_spectral_osc_test_target} generate_metal_osc)
    endif()
endforeach()

# Build every compiled CTest executable at once. All test exes are
# EXCLUDE_FROM_ALL, so a plain build skips them; CI uses this before ctest.
# The cmake -P source-scan tests (q_domain_contract, osc_backend_contract) need
# no build. Usage: cmake --build build --target tests_all && ctest --test-dir build
add_custom_target(tests_all)
add_dependencies(tests_all
    simulate
    arm_core_test
    core_contracts_test
    core_guarantees_test
    core_guarantees_drift_test
    osc_parity_test
    osc_width_parity_test
    q_ladder_parity_test
    q15_compute_precision_test
    q15_production_parity_test
    q15_simd_parity_test
    phase_nco_precision_test
    osc_recursive_test
    osc_sin_init_parity_test
    proc_mask_honesty_test
    full_fused_parity_test
    osc_formulas_domain_test
    window_backend_parity_test
    peak_interp_parabolic_test
    segment_endian_roundtrip_test
    segment_payload_bounds_test
    ifft_synth_parity_test
    daisy_config_layout_test
    convert_segments)

message(STATUS "Configured Spectral Engine")
message(STATUS "  desktop target: ${TARGET_DESKTOP}")
message(STATUS "  simulate target: ${TARGET_SIMULATION}")
message(STATUS "  simulate daisy target: ${TARGET_SIMULATION_DAISY}")
message(STATUS "  cuda enabled: ${SPECTRAL_USE_CUDA}")
if(NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "  build type: ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "  production build: ${SPECTRAL_PRODUCTION_BUILD}")
message(STATUS "  reproducible profile: ${SPECTRAL_REPRO_BUILD}")
message(STATUS "  debug logging: ${SPECTRAL_DEBUG_LOGGING}")
message(STATUS "  custom fast math mode: ${SPECTRAL_CUSTOM_FAST_MATH_MODE_STATUS}")
message(STATUS "  track pair omp chunk: ${SPECTRAL_TRACK_PAIR_OMP_CHUNK_CANONICAL}")
