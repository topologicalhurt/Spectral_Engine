# Guarantee-manifest / self-report harness (CTest), CAMPAIGN_2_MASTER_PLAN Phase B1/B2.
#
# Compiled tests for the active-guarantee bitset in core/spectral_guarantees.h
# (B2 self-report) and the per-gate error budgets in the guarantee registry
# (B1 manifest drift). The SAME source is built twice so the budgets are pinned
# against the actually-relaxed code path:
#
#   core_guarantees_test        - default gates. atan2/inv_sqrt/peak_log stay
#                                 exact (libm); TRIG is the one approximation that
#                                 is default-ON (degree-9 minimax sine), so its
#                                 EXACT_TRIG bit is cleared here too.
#   core_guarantees_drift_test  - SPECTRAL_ENABLE_APPROX_* all forced on: asserts
#                                 those bits are cleared in SPECTRAL_ACTIVE_GUARANTEES
#                                 AND each approximation stays within its budget.
#
# Run: cmake --build build --target core_guarantees_test core_guarantees_drift_test \
#      && ctest --test-dir build -R 'core_guarantees'

function(spectral_add_guarantees_test target)
    add_executable(${target} EXCLUDE_FROM_ALL
        "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_guarantees.c"
        "${SPECTRAL_CORE_DIR}/spectral_fast_math.c")
    spectral_apply_common_target(${target})
    target_link_libraries(${target} PRIVATE m)
    if(APPLE)
        target_link_options(${target} PRIVATE -Wl,-dead_strip)
    else()
        target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
        target_link_options(${target} PRIVATE -Wl,--gc-sections)
    endif()
endfunction()

# Default build: all exact guarantees active.
spectral_add_guarantees_test(core_guarantees_test)
add_test(NAME core_guarantees COMMAND core_guarantees_test)

# Relaxed build: force every approximation on so the drift budgets gate the
# actual approximate code paths (and the cleared bits exercise fail-closed).
spectral_add_guarantees_test(core_guarantees_drift_test)
target_compile_definitions(core_guarantees_drift_test PRIVATE
    SPECTRAL_ENABLE_APPROX_TRIG=1
    SPECTRAL_ENABLE_APPROX_ATAN2=1
    SPECTRAL_ENABLE_APPROX_INV_SQRT=1
    SPECTRAL_ENABLE_APPROX_PEAK_LOG=1)
add_test(NAME core_guarantees_drift COMMAND core_guarantees_drift_test)
