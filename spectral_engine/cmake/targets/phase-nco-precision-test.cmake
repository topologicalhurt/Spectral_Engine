# integer-NCO cubic phase precision characterization (CTest, Q5a).
#
# Measure-first gate for QTYPE_DOMAIN_PLAN.md Q5 (integer-NCO phase axis, the
# prerequisite for double-lane Q15 packing). Steps the spectral_phase_nco.h cubic
# forward-difference accumulator against the float-cubic phase and a double-precision
# truth over worst-case alpha/c2/c3/length segments, and reports (1) phase-index drift
# in LSBs and (2) the per-timbre dBFS cost of swapping the Q15 path's phase source from
# float-cubic to integer-NCO. Header-only inlines plus the test TU -- no production
# output changes; nothing linked from the engine.
#
# Run: cmake --build build --target phase_nco_precision_test \
#      && ctest --test-dir build -R phase_nco_precision

add_executable(phase_nco_precision_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_phase_nco_precision.c")

spectral_apply_common_target(phase_nco_precision_test)
target_link_libraries(phase_nco_precision_test PRIVATE m)

add_test(NAME phase_nco_precision COMMAND phase_nco_precision_test)
