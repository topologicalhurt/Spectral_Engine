# Q-domain contract harness (CTest).
#
# Source-scan (not compiled) guard for the Q0 type-discipline rule in
# QTYPE_DOMAIN_PLAN.md: float<->Q conversions go only through the named boundary
# macros (raw scale constants confined to their definition sites), and code in a
# // SPECTRAL_Q_DOMAIN BEGIN .. END region carries no float/double. The scan runs
# entirely in `cmake -P` — no toolchain dependency — so it works on every host.
#
# Run: ctest --test-dir build -R q_domain_contract

add_test(
    NAME q_domain_contract
    COMMAND ${CMAKE_COMMAND}
            -DSPECTRAL_Q_CONTRACT_ROOT=${SPECTRAL_ENGINE_ROOT}
            -P ${SPECTRAL_ENGINE_ROOT}/cmake/scripts/q_domain_contract.cmake)
