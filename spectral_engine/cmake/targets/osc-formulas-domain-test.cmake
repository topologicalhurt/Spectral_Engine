# Oscillator-formula domain/edge-case contract (CTest).
#
# Exercises the finiteness/range guards of core/spectral_osc_formulas.h (the
# SSOT waveform formulas shared by CPU/CUDA). Anchored on a confirmed
# signed-overflow UB at the spectral_osc_quantized INT_MAX boundary. Built
# under UBSan so a reintroduced overflow traps in addition to failing the
# behavioral assertions.
#
# Run: cmake --build build --target osc_formulas_domain_test
#      && ctest --test-dir build -R osc_formulas_domain

add_executable(osc_formulas_domain_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_osc_formulas_domain.c")

spectral_apply_common_target(osc_formulas_domain_test)
target_link_libraries(osc_formulas_domain_test PRIVATE m)

# UBSan: a reintroduced (int)scaled overflow aborts the test, not just changes
# a number. -fno-sanitize-recover makes the first UB fatal. Guarded to
# compilers that accept it; the behavioral asserts stand without it.
include(CheckCCompilerFlag)
check_c_compiler_flag("-fsanitize=undefined" SPECTRAL_HAS_UBSAN)
if(SPECTRAL_HAS_UBSAN)
    target_compile_options(osc_formulas_domain_test PRIVATE
        -fsanitize=undefined -fno-sanitize-recover=undefined)
    target_link_options(osc_formulas_domain_test PRIVATE -fsanitize=undefined)
endif()

add_test(NAME osc_formulas_domain COMMAND osc_formulas_domain_test)
