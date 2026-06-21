# Subtractive filter envelope contract (CTest) — RENDERER_ABSTRACTION_PLAN.md, Stage 3b.
#
# Pins the per-band spectral filter: gain interpolation + clamping + degenerate cases, and the
# per-partial apply (filtering as a spectral amplitude multiply). Fail-on-bug: break the
# interpolation or the bin->Hz mapping and a check fails. Links only the (pure, libm-only)
# envelope TU; its production caller (the subtractive renderer's source x filter) is Stage 3b-2.
#
# Run: cmake --build build --target filter_envelope_test \
#      && ctest --test-dir build -R filter_envelope

add_executable(filter_envelope_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_filter_envelope.c"
    "${SPECTRAL_CORE_DIR}/spectral_filter_envelope.c")

spectral_apply_common_target(filter_envelope_test)
target_link_libraries(filter_envelope_test PRIVATE m)
spectral_apply_dead_strip(filter_envelope_test)

add_test(NAME filter_envelope COMMAND filter_envelope_test)
