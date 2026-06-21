cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are disabled.\n"
        "Configure from repository root with:\n"
        "  cmake -S . -B build")
endif()

project(Spectral LANGUAGES C)

enable_testing()

add_subdirectory(spectral_engine)
