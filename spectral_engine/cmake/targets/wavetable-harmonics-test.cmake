# Wavetable frequency-domain bridge (CTest) — RENDERER_ABSTRACTION_PLAN.md, Stage 3a.
#
# Pins the harmonic extraction (DFT of the table samples) + the harmonic-expansion mapping
# against a synthetic table with known harmonics. Fail-on-bug: perturb the extraction scaling
# or the expansion bin/amp/phase mapping and a check fails. Links only the (pure, libm-only)
# bridge TU; it is not yet in the main source set (its production caller is Stage 3a-2).
#
# Run: cmake --build build --target wavetable_harmonics_test \
#      && ctest --test-dir build -R wavetable_harmonics

add_executable(wavetable_harmonics_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_wavetable_harmonics.c"
    "${SPECTRAL_CORE_DIR}/spectral_wavetable_harmonics.c")

spectral_apply_common_target(wavetable_harmonics_test)
target_link_libraries(wavetable_harmonics_test PRIVATE m)
spectral_apply_dead_strip(wavetable_harmonics_test)

add_test(NAME wavetable_harmonics COMMAND wavetable_harmonics_test)
