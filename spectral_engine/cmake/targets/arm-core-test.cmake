# ARM-core correctness harness (CTest).
#
# Exercises the REAL embedded synth (spectral_arm32_init/load/process) on the
# host with the M7 codepath forced on (SPECTRAL_ARM_M7=1) and the dual-MAC
# capability forced on (SPECTRAL_HAS_DUAL_MAC=1) so the voice-pairing path is
# covered too -- both run through the portable intrinsic fallbacks in
# spectral_q.h (host arm64 has no __ARM_FEATURE_DSP, and the portable smlald
# is bit-identical to the hardware SMLALD). Correctness half of CAMPAIGN_2_MASTER_PLAN A1b;
# the sim is the perf/resource model over the SAME real code.
#
# Run: cmake --build build --target arm_core_test && ctest --test-dir build -R arm32

add_executable(arm_core_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/arm_core/test_arm32_process.c"
    "${SPECTRAL_ARCH_ARM_DIR}/spectral_synth_arm32.c"
    "${SPECTRAL_CORE_DIR}/spectral_q.c"
    "${SPECTRAL_CORE_DIR}/spectral_lut.c")

spectral_apply_common_target(arm_core_test)
target_compile_definitions(arm_core_test PRIVATE
    SPECTRAL_EMBEDDED=1
    SPECTRAL_ARM_M7=1
    SPECTRAL_HAS_DUAL_MAC=1)
target_link_libraries(arm_core_test PRIVATE m)

# Drop the unused spectral_arm32_process_interleaved() (and its
# spectral_mono_to_stereo_q15 dependency in spectral_out.c) so the harness links
# against only arm32 + q15 + lut.
if(APPLE)
    target_link_options(arm_core_test PRIVATE -Wl,-dead_strip)
else()
    target_compile_options(arm_core_test PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(arm_core_test PRIVATE -Wl,--gc-sections)
endif()

add_test(NAME arm32_process_correctness COMMAND arm_core_test)
