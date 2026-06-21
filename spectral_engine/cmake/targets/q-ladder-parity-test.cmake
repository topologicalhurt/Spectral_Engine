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
# Built under UBSan (same house pattern as osc-formulas-domain-test.cmake): the
# widenings sit right on the C11 6.5.7p4 left-shift-of-negative trap (the min code
# is negative), so a reintroduced signed-domain shift -- or an overflowing
# narrowing bias-add -- aborts the run in addition to failing the behavioral
# assertions. The __int128 oracle is itself UB-free (multiplies by 2^k rather than
# shifting), so a trap can only come from the code under test.
#
# Run: cmake --build build --target q_ladder_parity_test \
#      && ctest --test-dir build -R q_ladder_parity

add_executable(q_ladder_parity_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_q_ladder_parity.c")

spectral_apply_common_target(q_ladder_parity_test)
target_link_libraries(q_ladder_parity_test PRIVATE m)

# UBSan: a reintroduced negative-shift in a widening or a signed overflow in a
# narrowing bias-add aborts the test, not just changes a number.
# -fno-sanitize-recover makes the first UB fatal. Guarded to compilers that accept
# it; the behavioral asserts stand without it.
include(CheckCCompilerFlag)
check_c_compiler_flag("-fsanitize=undefined" SPECTRAL_HAS_UBSAN)
if(SPECTRAL_HAS_UBSAN)
    target_compile_options(q_ladder_parity_test PRIVATE
        -fsanitize=undefined -fno-sanitize-recover=undefined)
    target_link_options(q_ladder_parity_test PRIVATE -fsanitize=undefined)
endif()

add_test(NAME q_ladder_parity COMMAND q_ladder_parity_test)
