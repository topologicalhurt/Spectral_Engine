# Segment endian-swap coverage contract (CTest).
#
# Asserts spectral_segment_swap_words covers the entire 64-byte Segment payload
# including the _pad_w cubic-MQ phase annotation, and is its own inverse.
# Regression for an endian swap that skipped _pad_w (cross-endian cubic-phase
# corruption when SPECTRAL_PRECISE_PHASE is enabled).
#
# Run: cmake --build build --target segment_endian_roundtrip_test
#      && ctest --test-dir build -R segment_endian_roundtrip

add_executable(segment_endian_roundtrip_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_segment_endian_roundtrip.c")

spectral_apply_common_target(segment_endian_roundtrip_test)
target_link_libraries(segment_endian_roundtrip_test PRIVATE m)

add_test(NAME segment_endian_roundtrip COMMAND segment_endian_roundtrip_test)
