# Wavetable Q15-format load de-quantization contract (CTest).
#
# Hand-builds a Q15-format .spwt and asserts the loaded samples carry the correct
# linear amplitude (spectral_sample_to_float(loaded) == q15/32768). Fails loudly if the
# cross-format (Q15-file -> float-runtime) branch ever widens the int16 without the
# /32768 scale again (the bin-0 = 16384 -> 16384.0 corruption).
#
# Run: cmake --build build --target wavetable_q15_load_test
#      && ctest --test-dir build -R wavetable_q15_load

add_executable(wavetable_q15_load_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_wavetable_q15_load.c"
    "${SPECTRAL_CORE_DIR}/spectral_wavetable.c"
    "${SPECTRAL_CORE_DIR}/spectral_fs.c"
    "${SPECTRAL_CORE_DIR}/spectral_log.c"
    "${SPECTRAL_CORE_DIR}/spectral_fast_math.c"
    "${SPECTRAL_RUNTIME_DIR}/spectral_utils.c")

spectral_apply_common_target(wavetable_q15_load_test)
target_link_libraries(wavetable_q15_load_test PRIVATE m)

add_test(NAME wavetable_q15_load COMMAND wavetable_q15_load_test)
