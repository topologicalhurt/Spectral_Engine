# Segment payload width-domain contract (CTest).
#
# Asserts spectral_segment_payload_valid upper-bounds |width| at
# SPECTRAL_SEGMENT_WIDTH_MAX (defense-in-depth for the rads*width
# int-conversion boundary behind the pass-248/249 INT_MAX defects), with an
# inclusive boundary and no over-tightening of the normal width domain.
#
# Run: cmake --build build --target segment_payload_bounds_test
#      && ctest --test-dir build -R segment_payload_bounds

add_executable(segment_payload_bounds_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_segment_payload_bounds.c")

spectral_apply_common_target(segment_payload_bounds_test)
target_link_libraries(segment_payload_bounds_test PRIVATE m)

add_test(NAME segment_payload_bounds COMMAND segment_payload_bounds_test)
