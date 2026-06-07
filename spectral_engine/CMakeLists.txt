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
    SPECTRAL_RESTRICTED_MODE=1
    SPECTRAL_SIMULATION_TARGET_DAISY=1)

# --- CACHE HEURISTICS ---
set(L3_SIZE_KB 0)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    execute_process(
        COMMAND sh -c "cat /sys/devices/system/cpu/cpu0/cache/index3/size 2>/dev/null | sed 's/K//'"
        OUTPUT_VARIABLE L3_SYSFS_OUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(L3_SYSFS_OUT MATCHES "^[0-9]+$")
        set(L3_SIZE_KB ${L3_SYSFS_OUT})
    endif()
elseif(APPLE)
    execute_process(
        COMMAND sysctl -n hw.l3cachesize
        OUTPUT_VARIABLE L3_SYSCTL_OUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(L3_SYSCTL_OUT MATCHES "^[0-9]+$")
        # hw.l3cachesize is in bytes
        math(EXPR L3_SIZE_KB "${L3_SYSCTL_OUT} / 1024")
    endif()
endif()

set(COMPUTED_STFT_CHUNK_FRAMES 512)
if(L3_SIZE_KB GREATER 0)
    # L3_SIZE_KB is e.g. 16384 (16MB). Target 50% of L3 cache to avoid thrashing, divided by 4 threads.
    # 50% of L3 (KB) = L3_SIZE_KB / 2. Per thread = L3_SIZE_KB / 8.
    # For a 4096 FFT, 2049 bins * 8 bytes (magsq+phase) = ~16 KB per frame.
    # frames = (L3_SIZE_KB / 8) / 16 = L3_SIZE_KB / 128
    math(EXPR COMPUTED_STFT_CHUNK_FRAMES "${L3_SIZE_KB} / 128")
endif()

# Clamp bounds (64 to 4096)
if(COMPUTED_STFT_CHUNK_FRAMES LESS 64)
    set(COMPUTED_STFT_CHUNK_FRAMES 64)
elseif(COMPUTED_STFT_CHUNK_FRAMES GREATER 4096)
    set(COMPUTED_STFT_CHUNK_FRAMES 4096)
endif()

add_compile_definitions(SPECTRAL_STFT_CHUNK_FRAMES=${COMPUTED_STFT_CHUNK_FRAMES})

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
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-compute-precision-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-production-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-simd-parity-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-throughput-bench.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/q15-pack8-bench.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/phase-nco-precision-test.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/full-fused-parity-test.cmake")

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
message(STATUS "  stft cache chunk frames: ${COMPUTED_STFT_CHUNK_FRAMES}")
