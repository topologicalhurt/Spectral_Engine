cmake_minimum_required(VERSION 3.21)

project(SpectralDaisySeed LANGUAGES C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

get_filename_component(SPECTRAL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SPECTRAL_ENGINE_ROOT "${SPECTRAL_REPO_ROOT}/spectral_engine")
include("${SPECTRAL_ENGINE_ROOT}/cmake/options.cmake")
include("${SPECTRAL_ENGINE_ROOT}/cmake/source-manifest.cmake")

spectral_daisy_validate_required_paths()
spectral_daisy_ensure_toolchain_tools()

file(MAKE_DIRECTORY "${SPECTRAL_DAISY_BIN_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SPECTRAL_DAISY_BIN_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${SPECTRAL_DAISY_BIN_DIR}")
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${SPECTRAL_DAISY_BIN_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} "${SPECTRAL_DAISY_BIN_DIR}")
endforeach()

set(SPECTRAL_DAISY_API_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/daisy_seed_spectral.c"
    ${SPECTRAL_SOURCES_DAISY_ENGINE})

spectral_daisy_collect_compile_definitions(SPECTRAL_DAISY_DEFS)
spectral_daisy_collect_include_dirs(SPECTRAL_DAISY_SDK_INCLUDE_DIRS)
spectral_daisy_collect_mcu_flags(SPECTRAL_DAISY_MCU_FLAGS)
spectral_daisy_collect_common_compile_options(SPECTRAL_DAISY_COMMON_COMPILE_OPTIONS)
spectral_daisy_collect_link_directories(SPECTRAL_DAISY_LINK_DIRS)
spectral_daisy_collect_link_libraries(SPECTRAL_DAISY_LINK_LIBS)
spectral_daisy_collect_link_options("${CMAKE_CURRENT_BINARY_DIR}/libspectral_daisy.map" SPECTRAL_DAISY_LINK_OPTIONS)

set(SPECTRAL_DAISY_API_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ${SPECTRAL_ENGINE_INCLUDE_DIRS}
    ${SPECTRAL_DAISY_SDK_INCLUDE_DIRS})

add_library(spectral_daisy_api_objects OBJECT ${SPECTRAL_DAISY_API_SOURCES})
target_include_directories(spectral_daisy_api_objects PRIVATE ${SPECTRAL_DAISY_API_INCLUDE_DIRS})
target_compile_definitions(spectral_daisy_api_objects PRIVATE ${SPECTRAL_DAISY_DEFS})
target_compile_options(spectral_daisy_api_objects PRIVATE
    ${SPECTRAL_DAISY_MCU_FLAGS}
    ${SPECTRAL_DAISY_COMMON_COMPILE_OPTIONS})

add_library(spectral_daisy_api STATIC $<TARGET_OBJECTS:spectral_daisy_api_objects>)
set_target_properties(spectral_daisy_api PROPERTIES OUTPUT_NAME spectral_daisy_api)
target_include_directories(spectral_daisy_api PUBLIC ${SPECTRAL_DAISY_API_INCLUDE_DIRS})
target_compile_definitions(spectral_daisy_api PUBLIC ${SPECTRAL_DAISY_DEFS})

add_executable(libspectral_daisy $<TARGET_OBJECTS:spectral_daisy_api_objects>)
set_target_properties(libspectral_daisy PROPERTIES
    OUTPUT_NAME libspectral_daisy
    SUFFIX ".elf"
    LINKER_LANGUAGE CXX)
target_link_directories(libspectral_daisy PRIVATE ${SPECTRAL_DAISY_LINK_DIRS})
target_link_options(libspectral_daisy PRIVATE ${SPECTRAL_DAISY_LINK_OPTIONS})
target_link_libraries(libspectral_daisy PRIVATE ${SPECTRAL_DAISY_LINK_LIBS})

add_custom_command(TARGET libspectral_daisy POST_BUILD
    COMMAND "${CMAKE_OBJCOPY}" -O ihex "$<TARGET_FILE:libspectral_daisy>" "${SPECTRAL_DAISY_BIN_DIR}/libspectral_daisy.hex"
    COMMAND "${CMAKE_OBJCOPY}" -O binary -S "$<TARGET_FILE:libspectral_daisy>" "${SPECTRAL_DAISY_BIN_DIR}/libspectral_daisy.bin"
    COMMAND "${CMAKE_SIZE}" "$<TARGET_FILE:libspectral_daisy>"
    VERBATIM)

add_custom_target(daisy_api DEPENDS libspectral_daisy)

add_custom_target(daisy_api_flash
    COMMAND "${SPECTRAL_DAISY_DFU_UTIL}" -a 0 -s 0x08000000:leave -D "${SPECTRAL_DAISY_BIN_DIR}/libspectral_daisy.bin"
    DEPENDS libspectral_daisy
    VERBATIM)

add_custom_target(daisy_api_info
    COMMAND "${CMAKE_COMMAND}" -E echo "Daisy API firmware target: ${SPECTRAL_DAISY_BIN_DIR}/libspectral_daisy.elf"
    COMMAND "${CMAKE_COMMAND}" -E echo "libDaisy: ${SPECTRAL_DAISY_LIBDAISY_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E echo "DaisySP: ${SPECTRAL_DAISY_DAISYSP_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E echo "Debug: ${SPECTRAL_DAISY_DEBUG} | Wavetable: ${SPECTRAL_DAISY_WAVETABLE}"
    VERBATIM)

if(SPECTRAL_DAISY_BUILD_EXAMPLE)
    add_subdirectory("${SPECTRAL_REPO_ROOT}/examples/daisy_seed" "${CMAKE_CURRENT_BINARY_DIR}/examples_daisy_seed")
endif()
