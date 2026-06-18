# Core-contracts correctness harness (CTest).
#
# Compiled (not string-grep) tests for the canonical kernel invariants declared in
# core/spectral_contracts.h. Currently exercises the COLA/WOLA overlap-add
# reconstruction invariant (CAMPAIGN_2_MASTER_PLAN Phase B0) against both a constructed
# periodic window and the engine's own spectral_window_* generators.
#
# Run: cmake --build build --target core_contracts_test && ctest --test-dir build -R core_contracts

add_executable(core_contracts_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_cola.c"
    "${SPECTRAL_CORE_DIR}/spectral_windows.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c")

spectral_apply_common_target(core_contracts_test)
# Force the portable (non-vDSP) window formulas so the COLA test exercises the
# source-visible window shapes deterministically and links no Accelerate fwk.
target_compile_definitions(core_contracts_test PRIVATE SPECTRAL_USE_VDSP=0)
target_link_libraries(core_contracts_test PRIVATE m)

# Strip the unused window descriptor/interp/peak machinery (and its fast-math
# dependency) so the harness links against only the window generators + the
# header-only contract predicates it actually calls.
if(APPLE)
    target_link_options(core_contracts_test PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(core_contracts_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(core_contracts_test PRIVATE -Wl,--gc-sections)
endif()

add_test(NAME core_contracts COMMAND core_contracts_test)
