# Full/fused analysis parity harness (CTest) — ULTRAPLAN Phase D, item D1.
#
# Compiled (not string-grep) realization of docs/core_audit/FULL_FUSED_PARITY_HARNESS.md.
# Drives the REAL production analysis entry point analyze_audio_with_path_mode()
# under PATH_FULL and PATH_FUSED over deterministic fixtures and asserts the two
# segment arrays agree within documented tolerances.
#
# Links the desktop engine sources (the genuine float analysis path) minus the
# CLI main(); the harness supplies its own main(). Mirrors targets/desktop.cmake
# so the analysis path — vDSP FFT, peak tracking, segment pooling — resolves
# exactly as in the shipped binary.
#
# Run: cmake --build build --target full_fused_parity_test \
#      && ctest --test-dir build -R full_fused_parity

set(SPECTRAL_PARITY_SOURCES ${SPECTRAL_SOURCES_TARGET_DESKTOP})
list(REMOVE_ITEM SPECTRAL_PARITY_SOURCES "${SPECTRAL_SOURCE_ENTRY_MAIN}")
list(APPEND SPECTRAL_PARITY_SOURCES
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_full_fused_parity.c")

if(APPLE)
    list(APPEND SPECTRAL_PARITY_SOURCES ${SPECTRAL_SOURCES_SYNTH_METAL})
    set_source_files_properties(${SPECTRAL_SOURCES_SYNTH_METAL} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
elseif(SPECTRAL_USE_CUDA)
    list(APPEND SPECTRAL_PARITY_SOURCES ${SPECTRAL_SOURCES_SYNTH_CUDA})
endif()

add_executable(full_fused_parity_test EXCLUDE_FROM_ALL ${SPECTRAL_PARITY_SOURCES})
spectral_apply_common_target(full_fused_parity_test)
target_link_libraries(full_fused_parity_test PRIVATE ${SPECTRAL_GPU_LINK_LIBS})

if(SPECTRAL_USE_CUDA)
    target_compile_definitions(full_fused_parity_test PRIVATE USE_CUDA)
    target_link_libraries(full_fused_parity_test PRIVATE CUDA::cudart)
    target_compile_options(full_fused_parity_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SPECTRAL_CUDA_COMPILE_OPTIONS}>)
    set_target_properties(full_fused_parity_test PROPERTIES LINKER_LANGUAGE C)
endif()

add_test(NAME full_fused_parity COMMAND full_fused_parity_test)
