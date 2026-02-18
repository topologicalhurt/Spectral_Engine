# Canonical home for Spectral engine build options and cache variables.

if(DEFINED SPECTRAL_OPTIONS_INCLUDED)
    return()
endif()
set(SPECTRAL_OPTIONS_INCLUDED TRUE)

if(NOT DEFINED SPECTRAL_ENGINE_ROOT OR NOT DEFINED SPECTRAL_REPO_ROOT)
    message(FATAL_ERROR
        "spectral_engine/cmake/options.cmake requires SPECTRAL_ENGINE_ROOT "
        "and SPECTRAL_REPO_ROOT to be defined before inclusion.")
endif()

option(SPECTRAL_SKIP_VERSION_CHECK "Skip minimum compiler version checks" OFF)
option(SPECTRAL_REPRO_BUILD "Use reproducible optimization profile" ON)

set(SPECTRAL_PGO "off" CACHE STRING "Profile-guided optimization mode (off, gen, use)")
set_property(CACHE SPECTRAL_PGO PROPERTY STRINGS off gen use)
set(SPECTRAL_PGO_DIR "${SPECTRAL_REPO_ROOT}/output/pgo" CACHE PATH "Profile-guided optimization directory")
set(SPECTRAL_PGO_PROFILE "${SPECTRAL_PGO_DIR}/spectral.profdata" CACHE FILEPATH "Clang PGO profile data file")

set(SPECTRAL_SIMULATION_BOARD "generic" CACHE STRING "Simulation profile board (generic or daisy)")
set_property(CACHE SPECTRAL_SIMULATION_BOARD PROPERTY STRINGS generic daisy)

set(SPECTRAL_VALID_PGO_MODES off gen use)
if(NOT SPECTRAL_PGO IN_LIST SPECTRAL_VALID_PGO_MODES)
    message(FATAL_ERROR "Invalid SPECTRAL_PGO='${SPECTRAL_PGO}'. Use off, gen, or use.")
endif()

set(SPECTRAL_VALID_SIMULATION_BOARDS generic daisy)
if(NOT SPECTRAL_SIMULATION_BOARD IN_LIST SPECTRAL_VALID_SIMULATION_BOARDS)
    message(FATAL_ERROR "Invalid SPECTRAL_SIMULATION_BOARD='${SPECTRAL_SIMULATION_BOARD}'. Use generic or daisy.")
endif()

set(SPECTRAL_BENCH_SCRIPT "${SPECTRAL_REPO_ROOT}/tools/benchmark_spectral.sh" CACHE FILEPATH "Benchmark harness script")
set(SPECTRAL_BENCH_INPUT "${SPECTRAL_REPO_ROOT}/resources/motormouth_recites_shakespeare_he_saw_the_cat.wav" CACHE FILEPATH "Benchmark input file")
set(SPECTRAL_BENCH_RUNS "6" CACHE STRING "Benchmark run count")
set(SPECTRAL_BENCH_ARGS "0 1.0 0 4096 128 -90 8 1" CACHE STRING "Benchmark CLI arguments")
set(SPECTRAL_BENCH_CACHE_ARGS "${SPECTRAL_BENCH_ARGS}" CACHE STRING "Benchmark cache-mode CLI arguments")

set(SPECTRAL_CUDA_AVAILABLE OFF)
find_program(SPECTRAL_NVCC_EXECUTABLE nvcc)
if(SPECTRAL_NVCC_EXECUTABLE)
    set(SPECTRAL_CUDA_AVAILABLE ON)
endif()

if(APPLE)
    set(SPECTRAL_USE_CUDA_DEFAULT OFF)
else()
    set(SPECTRAL_USE_CUDA_DEFAULT ${SPECTRAL_CUDA_AVAILABLE})
endif()
option(SPECTRAL_USE_CUDA "Enable CUDA backend when available (Linux only)" ${SPECTRAL_USE_CUDA_DEFAULT})

if(SPECTRAL_USE_CUDA AND NOT SPECTRAL_CUDA_AVAILABLE)
    message(WARNING "SPECTRAL_USE_CUDA=ON but nvcc was not found; disabling CUDA.")
    set(SPECTRAL_USE_CUDA OFF CACHE BOOL "Enable CUDA backend when available (Linux only)" FORCE)
endif()

set(SPECTRAL_DAISY_SOURCE_DIR "${SPECTRAL_REPO_ROOT}/api/daisy_seed" CACHE PATH "Daisy CMake source directory")
set(SPECTRAL_DAISY_BUILD_DIR "${SPECTRAL_REPO_ROOT}/build/daisy" CACHE PATH "Daisy CMake build directory")
set(SPECTRAL_DAISY_BIN_DIR "${SPECTRAL_REPO_ROOT}/build/bin/daisy" CACHE PATH "Daisy output artifact directory")
set(SPECTRAL_DAISY_TOOLCHAIN_FILE "${SPECTRAL_ENGINE_ROOT}/cmake/toolchains/arm-none-eabi-gcc.cmake" CACHE FILEPATH "Daisy ARM GCC CMake toolchain file")
set(SPECTRAL_DAISY_BUILD_EXAMPLE OFF CACHE BOOL "Build Daisy example firmware target")

set(_SPECTRAL_DAISY_PATH_DEFAULT "$ENV{DAISY_PATH}")
if(NOT "${_SPECTRAL_DAISY_PATH_DEFAULT}" STREQUAL "")
    set(_SPECTRAL_DAISY_LIBDAISY_DEFAULT "${_SPECTRAL_DAISY_PATH_DEFAULT}/libDaisy")
    set(_SPECTRAL_DAISY_DAISYSP_DEFAULT "${_SPECTRAL_DAISY_PATH_DEFAULT}/DaisySP")
else()
    set(_SPECTRAL_DAISY_LIBDAISY_DEFAULT "$ENV{HOME}/daisy/libDaisy")
    set(_SPECTRAL_DAISY_DAISYSP_DEFAULT "$ENV{HOME}/daisy/DaisySP")
endif()

set(SPECTRAL_DAISY_LIBDAISY_DIR "${_SPECTRAL_DAISY_LIBDAISY_DEFAULT}" CACHE PATH "Path to libDaisy")
set(SPECTRAL_DAISY_DAISYSP_DIR "${_SPECTRAL_DAISY_DAISYSP_DEFAULT}" CACHE PATH "Path to DaisySP")
set(SPECTRAL_DAISY_TOOLCHAIN_PREFIX "arm-none-eabi-" CACHE STRING "ARM GCC toolchain prefix")
set(SPECTRAL_DAISY_CPU "-mcpu=cortex-m7" CACHE STRING "Daisy CPU flag")
set(SPECTRAL_DAISY_FPU "-mfpu=fpv5-d16" CACHE STRING "Daisy FPU flag")
set(SPECTRAL_DAISY_FLOAT_ABI "-mfloat-abi=hard" CACHE STRING "Daisy float ABI flag")
set(SPECTRAL_DAISY_DFU_UTIL "dfu-util" CACHE STRING "DFU utility executable")
set(SPECTRAL_DAISY_LDSCRIPT "${SPECTRAL_DAISY_LIBDAISY_DIR}/core/STM32H750IB_flash.lds" CACHE FILEPATH "Daisy linker script")

set(SPECTRAL_DAISY_DEBUG OFF CACHE BOOL "Enable Daisy debug defines and symbols")
set(SPECTRAL_DAISY_WAVETABLE OFF CACHE BOOL "Enable wavetable LUT support on Daisy")
set(SPECTRAL_DAISY_SAFE_MATH ON CACHE BOOL "Disable unsafe fast-math flags for deterministic Daisy builds")
set(SPECTRAL_DAISY_OPTIMIZE "3" CACHE STRING "Optimization level for Daisy C sources")
set(SPECTRAL_DAISY_EXAMPLE_OPT_DEBUG "-Og -g" CACHE STRING "Debug optimization flags for Daisy example C++")
set(SPECTRAL_DAISY_EXAMPLE_OPT_RELEASE "-O2" CACHE STRING "Release optimization flags for Daisy example C++")

set(SPECTRAL_DAISY_BLINK_PLAYING "" CACHE STRING "LED blink period (ms) while playing")
set(SPECTRAL_DAISY_BLINK_DONE "" CACHE STRING "LED blink period (ms) when playback is complete")
set(SPECTRAL_DAISY_ERROR_BLINK_ON "" CACHE STRING "Error blink ON period (ms)")
set(SPECTRAL_DAISY_ERROR_BLINK_OFF "" CACHE STRING "Error blink OFF period (ms)")

set(SPECTRAL_DAISY_FORWARD_CACHE_VARS
    SPECTRAL_DAISY_BIN_DIR
    SPECTRAL_DAISY_BUILD_EXAMPLE
    SPECTRAL_DAISY_LIBDAISY_DIR
    SPECTRAL_DAISY_DAISYSP_DIR
    SPECTRAL_DAISY_TOOLCHAIN_PREFIX
    SPECTRAL_DAISY_CPU
    SPECTRAL_DAISY_FPU
    SPECTRAL_DAISY_FLOAT_ABI
    SPECTRAL_DAISY_DFU_UTIL
    SPECTRAL_DAISY_LDSCRIPT
    SPECTRAL_DAISY_DEBUG
    SPECTRAL_DAISY_WAVETABLE
    SPECTRAL_DAISY_SAFE_MATH
    SPECTRAL_DAISY_OPTIMIZE
    SPECTRAL_DAISY_EXAMPLE_OPT_DEBUG
    SPECTRAL_DAISY_EXAMPLE_OPT_RELEASE
    SPECTRAL_DAISY_BLINK_PLAYING
    SPECTRAL_DAISY_BLINK_DONE
    SPECTRAL_DAISY_ERROR_BLINK_ON
    SPECTRAL_DAISY_ERROR_BLINK_OFF)

include("${CMAKE_CURRENT_LIST_DIR}/daisy-config.cmake")
