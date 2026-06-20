# Error-origin logging contract (CTest) — kernel-hardening V.4 / AI_CANON #29.
#
# analyze_audio's defensive early returns (null argument, invalid analysis shape) are
# library-boundary origins the CLI can't reach — it pre-validates its args — so they are
# pinned here against the REAL analyze_audio entry point: each must emit its structured
# error log. Fail-on-bug: revert either spectral_log_error_codef call in spectral_analysis.c
# to a silent return and this test fails.
#
# Links the desktop engine sources minus the CLI main() (the harness supplies its own),
# mirroring full-fused-parity-test.cmake so analyze_audio resolves exactly as shipped.
#
# Run: cmake --build build --target error_origin_logs_test \
#      && ctest --test-dir build -R error_origin_logs

set(SPECTRAL_ERRLOG_SOURCES ${SPECTRAL_SOURCES_TARGET_DESKTOP})
list(REMOVE_ITEM SPECTRAL_ERRLOG_SOURCES "${SPECTRAL_SOURCE_ENTRY_MAIN}")
list(APPEND SPECTRAL_ERRLOG_SOURCES
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_error_origin_logs.c")

if(APPLE)
    list(APPEND SPECTRAL_ERRLOG_SOURCES ${SPECTRAL_SOURCES_SYNTH_METAL})
    set_source_files_properties(${SPECTRAL_SOURCES_SYNTH_METAL} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
elseif(SPECTRAL_USE_CUDA)
    list(APPEND SPECTRAL_ERRLOG_SOURCES ${SPECTRAL_SOURCES_SYNTH_CUDA})
endif()

add_executable(error_origin_logs_test EXCLUDE_FROM_ALL ${SPECTRAL_ERRLOG_SOURCES})
spectral_apply_common_target(error_origin_logs_test)
target_link_libraries(error_origin_logs_test PRIVATE ${SPECTRAL_GPU_LINK_LIBS})

if(SPECTRAL_USE_CUDA)
    target_compile_definitions(error_origin_logs_test PRIVATE USE_CUDA)
    target_link_libraries(error_origin_logs_test PRIVATE CUDA::cudart)
    target_compile_options(error_origin_logs_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SPECTRAL_CUDA_COMPILE_OPTIONS}>)
    set_target_properties(error_origin_logs_test PROPERTIES LINKER_LANGUAGE C)
endif()

add_test(NAME error_origin_logs COMMAND error_origin_logs_test)
