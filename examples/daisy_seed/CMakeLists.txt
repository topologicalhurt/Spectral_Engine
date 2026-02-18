cmake_minimum_required(VERSION 3.21)

if(NOT TARGET spectral_daisy_api)
    message(FATAL_ERROR "examples/daisy_seed must be configured through api/daisy_seed.")
endif()

get_filename_component(SPECTRAL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SPECTRAL_ENGINE_ROOT "${SPECTRAL_REPO_ROOT}/spectral_engine")
include("${SPECTRAL_ENGINE_ROOT}/cmake/options.cmake")
include("${SPECTRAL_ENGINE_ROOT}/cmake/source-manifest.cmake")

spectral_daisy_ensure_toolchain_tools()

spectral_daisy_collect_compile_definitions(SPECTRAL_DAISY_DEFS)
spectral_daisy_collect_include_dirs(SPECTRAL_DAISY_SDK_INCLUDE_DIRS)
spectral_daisy_collect_mcu_flags(SPECTRAL_DAISY_MCU_FLAGS)
spectral_daisy_collect_common_compile_options(SPECTRAL_DAISY_COMMON_COMPILE_OPTIONS)
spectral_daisy_collect_example_cxx_options(SPECTRAL_DAISY_EXAMPLE_CXX_OPTIONS)
spectral_daisy_collect_link_directories(SPECTRAL_DAISY_LINK_DIRS)
spectral_daisy_collect_link_libraries(SPECTRAL_DAISY_LINK_LIBS)
spectral_daisy_collect_link_options("${CMAKE_CURRENT_BINARY_DIR}/spectral_example.map" SPECTRAL_DAISY_LINK_OPTIONS)

add_executable(spectral_example "${CMAKE_CURRENT_SOURCE_DIR}/example_spectral.cpp")
set_target_properties(spectral_example PROPERTIES
    OUTPUT_NAME spectral_example
    SUFFIX ".elf"
    LINKER_LANGUAGE CXX)

target_include_directories(spectral_example PRIVATE
    "${SPECTRAL_REPO_ROOT}/api/daisy_seed"
    ${SPECTRAL_ENGINE_INCLUDE_DIRS}
    ${SPECTRAL_DAISY_SDK_INCLUDE_DIRS})

target_compile_definitions(spectral_example PRIVATE ${SPECTRAL_DAISY_DEFS})
target_compile_options(spectral_example PRIVATE
    ${SPECTRAL_DAISY_MCU_FLAGS}
    ${SPECTRAL_DAISY_COMMON_COMPILE_OPTIONS}
    $<$<COMPILE_LANGUAGE:CXX>:${SPECTRAL_DAISY_EXAMPLE_CXX_OPTIONS}>)

target_link_directories(spectral_example PRIVATE ${SPECTRAL_DAISY_LINK_DIRS})
target_link_options(spectral_example PRIVATE ${SPECTRAL_DAISY_LINK_OPTIONS})
target_link_libraries(spectral_example PRIVATE spectral_daisy_api ${SPECTRAL_DAISY_LINK_LIBS})

add_custom_command(TARGET spectral_example POST_BUILD
    COMMAND "${CMAKE_OBJCOPY}" -O ihex "$<TARGET_FILE:spectral_example>" "${SPECTRAL_DAISY_BIN_DIR}/spectral_example.hex"
    COMMAND "${CMAKE_OBJCOPY}" -O binary -S "$<TARGET_FILE:spectral_example>" "${SPECTRAL_DAISY_BIN_DIR}/spectral_example.bin"
    COMMAND "${CMAKE_SIZE}" "$<TARGET_FILE:spectral_example>"
    VERBATIM)

add_custom_target(daisy_example DEPENDS spectral_example)

add_custom_target(daisy_example_flash
    COMMAND "${SPECTRAL_DAISY_DFU_UTIL}" -a 0 -s 0x08000000:leave -D "${SPECTRAL_DAISY_BIN_DIR}/spectral_example.bin"
    DEPENDS spectral_example
    VERBATIM)
