# Public 1.0 API freeze guard (CTest) — spectral_kernel.h.
#
# Pins the frozen 1.0 surface: the version macros, the scene-model struct layouts, and the
# signatures of the curated public entry points (typed function-pointer assignments — a drifted
# signature fails to compile). Links the desktop engine sources minus the CLI main() (the freeze
# test references symbols across analysis/synth/io/renderer/wavetable), mirroring
# full-fused-parity-test.cmake.
#
# Run: cmake --build build --target kernel_api_freeze_test \
#      && ctest --test-dir build -R kernel_api_freeze

set(SPECTRAL_KERNEL_FREEZE_SOURCES ${SPECTRAL_SOURCES_TARGET_DESKTOP})
list(REMOVE_ITEM SPECTRAL_KERNEL_FREEZE_SOURCES "${SPECTRAL_SOURCE_ENTRY_MAIN}")
list(APPEND SPECTRAL_KERNEL_FREEZE_SOURCES
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_kernel_api_freeze.c")

if(APPLE)
    list(APPEND SPECTRAL_KERNEL_FREEZE_SOURCES ${SPECTRAL_SOURCES_SYNTH_METAL})
    set_source_files_properties(${SPECTRAL_SOURCES_SYNTH_METAL} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
elseif(SPECTRAL_USE_CUDA)
    list(APPEND SPECTRAL_KERNEL_FREEZE_SOURCES ${SPECTRAL_SOURCES_SYNTH_CUDA})
endif()

add_executable(kernel_api_freeze_test EXCLUDE_FROM_ALL ${SPECTRAL_KERNEL_FREEZE_SOURCES})
spectral_apply_common_target(kernel_api_freeze_test)
target_link_libraries(kernel_api_freeze_test PRIVATE ${SPECTRAL_GPU_LINK_LIBS})

if(SPECTRAL_USE_CUDA)
    target_compile_definitions(kernel_api_freeze_test PRIVATE USE_CUDA)
    target_link_libraries(kernel_api_freeze_test PRIVATE CUDA::cudart)
    target_compile_options(kernel_api_freeze_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SPECTRAL_CUDA_COMPILE_OPTIONS}>)
    set_target_properties(kernel_api_freeze_test PROPERTIES LINKER_LANGUAGE C)
endif()

add_test(NAME kernel_api_freeze COMMAND kernel_api_freeze_test)
