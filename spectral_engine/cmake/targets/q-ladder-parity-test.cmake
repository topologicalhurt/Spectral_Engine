# Signed Q-ladder conversion parity harness (CTest).
#
# Gates the six directed conversions of the signed fixed-point ladder in
# spectral_q.h -- q15<->q31<->q63 -- each against an integer-exact (__int128)
# oracle: exhaustive over all 65536 q15 on the widen/round-trip side, a directed
# boundary table (both saturation bands + the +-0.5 LSB ties) and a dense random
# sweep on the narrow side. Tolerance is EXACT (==) throughout: widening and
# round-trips are bit-identities, each narrowing is bit-exact vs the oracle's
# round-half-up + saturate. Header-only -- pulls spectral_q.h and the shared RNG;
# nothing from the engine is linked.
#
# Run: cmake --build build --target q_ladder_parity_test \
#      && ctest --test-dir build -R q_ladder_parity

add_executable(q_ladder_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_q_ladder_parity.c")

spectral_apply_common_target(q_ladder_parity_test)
target_link_libraries(q_ladder_parity_test PRIVATE m)

add_test(NAME q_ladder_parity COMMAND q_ladder_parity_test)
