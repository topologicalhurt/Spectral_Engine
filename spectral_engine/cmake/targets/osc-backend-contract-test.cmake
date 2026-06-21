# Oscillator backend-contract harness (CTest).
#
# Source-scan (not compiled) gate for the OSCILLATOR_BACKEND_CONTRACT_PLAN.md Phase-3
# anti-drift rule: every backend that RE-IMPLEMENTS the canonical Q15 waveform contract
# (i.e. #includes spectral_osc_q15.h) must carry a _Static_assert version pin, so adding a
# Q15 backend without acknowledging the contract version fails CI. Pin VALUE-correctness is
# the compiler's job (the _Static_assert itself); this scan owns pin PRESENCE, which the
# compiler cannot see. Consumers are DISCOVERED by glob, not hard-coded. The scan runs
# entirely in `cmake -P` -- no toolchain dependency -- so it works on every host (and needs
# neither the GPU nor real Cortex-M hardware the backends themselves require).
#
# Run: ctest --test-dir build -R osc_backend_contract

add_test(
    NAME osc_backend_contract
    COMMAND ${CMAKE_COMMAND}
            -DSPECTRAL_OSC_CONTRACT_ROOT=${SPECTRAL_ENGINE_ROOT}
            -P ${SPECTRAL_ENGINE_ROOT}/cmake/scripts/osc_backend_contract.cmake)
