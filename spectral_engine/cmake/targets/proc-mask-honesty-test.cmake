# Processing-chain honesty contract (CTest).
#
# Pins that spectral_process_chain_apply never silently no-ops a requested stage:
# a reserved (unimplemented) stage and the experimental adaptive_track_density
# (when its SPECTRAL_PRECISE_PHASE build gate is off) both fail loudly, while
# mask=NONE is a true no-op. Regression gate for the SD-5 / W5 honesty fixes.
#
# Run: cmake --build build --target proc_mask_honesty_test \
#      && ctest --test-dir build -R proc_mask_honesty

add_executable(proc_mask_honesty_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_proc_mask_honesty.c"
    "${SPECTRAL_ANALYSIS_DIR}/spectral_processing_chain.c"
    "${SPECTRAL_ANALYSIS_DIR}/spectral_proc_adaptive_track_density.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c"
    "${SPECTRAL_CORE_DIR}/spectral_log.c")

spectral_apply_common_target(proc_mask_honesty_test)
target_link_libraries(proc_mask_honesty_test PRIVATE m)

if(APPLE)
    target_link_options(proc_mask_honesty_test PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(proc_mask_honesty_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(proc_mask_honesty_test PRIVATE -Wl,--gc-sections)
endif()

add_test(NAME proc_mask_honesty COMMAND proc_mask_honesty_test)
