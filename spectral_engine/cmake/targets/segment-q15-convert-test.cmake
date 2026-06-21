# float->Q15 runtime segment conversion faithfulness contract (CTest).
#
# Pins that spectral_segment_to_q15_runtime REJECTS a grain whose stretched length exceeds
# the uint16 length field (<= 65535 output samples) instead of silently truncating the
# partial mid-note, and that lengths up to the limit still convert exactly.
#
# Run: cmake --build build --target segment_q15_convert_test
#      && ctest --test-dir build -R segment_q15_convert

add_executable(segment_q15_convert_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_segment_q15_convert.c"
    "${SPECTRAL_CORE_DIR}/spectral_segment_convert.c")

spectral_apply_common_target(segment_q15_convert_test)
target_link_libraries(segment_q15_convert_test PRIVATE m)

add_test(NAME segment_q15_convert COMMAND segment_q15_convert_test)
