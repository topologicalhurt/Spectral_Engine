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

set(SPECTRAL_BIN_DIR "${SPECTRAL_REPO_ROOT}/build/bin")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SPECTRAL_BIN_DIR}")
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${SPECTRAL_BIN_DIR}")
endforeach()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/source-manifest.cmake")

set(SPECTRAL_XXHASH_SOURCE "${SPECTRAL_REPO_ROOT}/third_party/xxHash/xxhash.c")
add_library(spectral_xxhash STATIC "${SPECTRAL_XXHASH_SOURCE}")
target_include_directories(spectral_xxhash PUBLIC "${SPECTRAL_XXHASH_INCLUDE_DIR}")

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

set(SPECTRAL_EMBEDDED_FLOAT_DEFINES SPECTRAL_EMBEDDED_FLOAT=1)
set(SPECTRAL_EMBEDDED_RESTRICTED_DEFINES SPECTRAL_RESTRICTED_MODE=1 SPECTRAL_NO_PERF=1)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/host-config.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/desktop.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/simulation.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/embedded.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/cuda.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/convert-segments.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/daisy.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/utilities.cmake")

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
