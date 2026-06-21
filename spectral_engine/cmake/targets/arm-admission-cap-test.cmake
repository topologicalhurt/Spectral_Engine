# ARM-core admission-cap gate (CTest).
#
# Pins the per-block active-voice admission cap (SPECTRAL_ARM32_ACTIVE_CAP) in the
# REAL embedded synth (spectral_arm32_init/load/process), built on the host with
# the M7 codepath forced on. The cap is forced LOW here (8, below the 512 storage
# bound) so it binds: the fixture overlaps more voices than the cap and asserts
# the kernel admits exactly the cap, with the surplus deferred (backpressure).
#
# Distinct from arm_core_test (which runs at the default cap = storage); this is
# the only coverage that the WORK bound is decoupled from the STORAGE bound.
#
# Run: cmake --build build --target arm_admission_cap_test
#      && ctest --test-dir build -R arm32_admission_cap

add_executable(arm_admission_cap_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/arm_core/test_arm32_admission_cap.c"
    "${SPECTRAL_ARCH_ARM_DIR}/spectral_synth_arm32.c"
    "${SPECTRAL_CORE_DIR}/spectral_q.c"
    "${SPECTRAL_CORE_DIR}/spectral_lut.c")

spectral_apply_common_target(arm_admission_cap_test)
target_compile_definitions(arm_admission_cap_test PRIVATE
    SPECTRAL_EMBEDDED=1
    SPECTRAL_ARM_M7=1
    SPECTRAL_HAS_DUAL_MAC=1
    SPECTRAL_ARM32_ACTIVE_CAP=8)
target_link_libraries(arm_admission_cap_test PRIVATE m)

# Firmware-faithful, deterministic codegen (same rationale as arm-core-test).
target_compile_options(arm_admission_cap_test PRIVATE -fno-fast-math -fno-lto)
target_link_options(arm_admission_cap_test PRIVATE -fno-lto)

spectral_apply_dead_strip(arm_admission_cap_test)

add_test(NAME arm32_admission_cap COMMAND arm_admission_cap_test)
