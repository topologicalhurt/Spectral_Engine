# Oscillator backend-contract checker (run as: cmake -DSPECTRAL_OSC_CONTRACT_ROOT=<dir> -P this).
#
# Closes the one anti-drift hole the per-consumer _Static_assert pins leave open:
# the pins themselves are OPT-IN, so a NEW Q15 backend can re-implement the canonical
# waveform contract and silently forget to pin its version. This source scan makes the
# matrix self-enforcing — "add a Q15 backend without acknowledging the contract version
# and CI fails" — the OSCILLATOR_BACKEND_CONTRACT_PLAN.md Phase-3 deliverable.
#
# Two-part anti-drift, by design (this scan owns only the second part):
#
#   (A) VALUE correctness — owned by the COMPILER. Each Q15 re-implementer carries a
#       _Static_assert(SPECTRAL_OSC_Q15_VERSION == N, ...). Bump the contract version in
#       spectral_osc_q15.h and every consumer's `== N` stops holding -> the build fails
#       until each is re-validated. Nothing for this scan to add there.
#
#   (B) PIN PRESENCE — owned by THIS scan. The compiler can only check a pin that EXISTS;
#       it cannot notice a re-implementer that never wrote one. So:
#
#         RULE Q15 — every file that `#include "spectral_osc_q15.h"` (other than the
#         header itself) is a re-implementer of the canonical Q15 waveform contract and
#         MUST contain a `_Static_assert(... SPECTRAL_OSC_Q15_VERSION == ...)` pin. The
#         consumer set is DISCOVERED (glob), not hard-coded, so a brand-new backend that
#         includes the contract is automatically held to the rule. Today's three:
#         core/spectral_oscillator.c (scalar), arch/simd/oscillator_simd.c (SIMDe pack8),
#         arch/arm/oscillator_simd.c (CMSIS).
#
# WHY ONLY Q15 GETS A PIN RULE (and float does not): the Q15 waveform is RE-IMPLEMENTED
# per backend, so independent copies can drift -> each needs a version handshake. The
# float contract spectral_osc_formulas.h is a SHARED `static inline` source of truth that
# callers merely CALL — there is nothing to drift from for a plain includer. The float
# backends that genuinely re-implement are gated elsewhere and intentionally NOT by this
# scan: SIMDe by the osc_parity / osc_width_parity RUNTIME CTests (numeric parity, stronger
# than a version token), GPU-Metal by verify_metal_osc (the MSL is codegen'd from the C
# formulas; the build fails on drift), CUDA likewise from the shared evaluator. This scan
# only sanity-asserts BOTH contracts still declare a numeric version, so the matrix header
# in spectral_oscillator_dispatch.h never silently loses its anchor.
#
# Implementation note: presence is a whole-file REGEX MATCH (not a per-line walk) — we are
# asserting a token EXISTS somewhere, not reporting a line, so the q_domain_contract-style
# manual line walk is unnecessary here. `[^;]*` keeps the match inside one _Static_assert
# (its terminating `;`), so an unrelated assert above the Q15 one cannot satisfy the rule.
#
# A FATAL_ERROR (nonzero exit) fails the osc_backend_contract CTest.

if(NOT DEFINED SPECTRAL_OSC_CONTRACT_ROOT)
    message(FATAL_ERROR "osc_backend_contract: SPECTRAL_OSC_CONTRACT_ROOT not set")
endif()

set(_q15_header   "spectral_osc_q15.h")
set(_float_header "spectral_osc_formulas.h")

file(GLOB_RECURSE _sources
    "${SPECTRAL_OSC_CONTRACT_ROOT}/*.c"
    "${SPECTRAL_OSC_CONTRACT_ROOT}/*.h"
    "${SPECTRAL_OSC_CONTRACT_ROOT}/*.cu")

set(_violations "")
set(_q15_version "")
set(_float_version "")
set(_q15_header_path "")
set(_consumers "")

# --- Pass 1: locate the contract headers and read their declared versions ----------
foreach(_file IN LISTS _sources)
    get_filename_component(_base "${_file}" NAME)
    if(_base STREQUAL _q15_header)
        set(_q15_header_path "${_file}")
        file(READ "${_file}" _hc)
        string(REGEX MATCH "#define[ \t]+SPECTRAL_OSC_Q15_VERSION[ \t]+([0-9]+)" _m "${_hc}")
        if(NOT _m STREQUAL "")
            set(_q15_version "${CMAKE_MATCH_1}")
        endif()
    elseif(_base STREQUAL _float_header)
        file(READ "${_file}" _hc)
        string(REGEX MATCH "#define[ \t]+SPECTRAL_OSC_FORMULAS_VERSION[ \t]+([0-9]+)" _m "${_hc}")
        if(NOT _m STREQUAL "")
            set(_float_version "${CMAKE_MATCH_1}")
        endif()
    endif()
endforeach()

if(_q15_version STREQUAL "")
    list(APPEND _violations
        "  ${_q15_header}: SPECTRAL_OSC_Q15_VERSION is not defined (the Q15 waveform contract lost its version anchor)")
endif()
if(_float_version STREQUAL "")
    list(APPEND _violations
        "  ${_float_header}: SPECTRAL_OSC_FORMULAS_VERSION is not defined (the float waveform contract lost its version anchor)")
endif()

# --- Pass 2: every Q15-contract includer must carry a version-pin _Static_assert ----
foreach(_file IN LISTS _sources)
    if(_file STREQUAL _q15_header_path)
        continue()
    endif()
    file(READ "${_file}" _content)
    string(REPLACE "\r" "" _content "${_content}")

    string(FIND "${_content}" "#include \"${_q15_header}\"" _inc)
    if(_inc EQUAL -1)
        continue()
    endif()

    list(APPEND _consumers "${_file}")

    # A _Static_assert whose comparison mentions the contract version, inside one
    # assert (the `[^;]*` cannot cross the terminating `;`). Tolerates both the direct
    # form (`== 1`) and the named-pin form (`== SPECTRAL_OSC_Q15_VERSION_CMSIS_PIN`).
    string(REGEX MATCH
        "_Static_assert[^;]*SPECTRAL_OSC_Q15_VERSION[ \t\r\n]*=="
        _pin "${_content}")
    if(_pin STREQUAL "")
        list(APPEND _violations
            "  ${_file}: includes the canonical Q15 contract (${_q15_header}) but carries no _Static_assert(... SPECTRAL_OSC_Q15_VERSION == ...) pin -- a Q15 re-implementer must version-handshake the contract so a bump forces re-validation here. See SPECTRAL_OSC_Q15_VERSION in ${_q15_header} and the existing pins in spectral_oscillator.c / arch/simd/oscillator_simd.c / arch/arm/oscillator_simd.c.")
    endif()
endforeach()

list(LENGTH _violations _nv)
if(_nv GREATER 0)
    string(REPLACE ";" "\n" _report "${_violations}")
    message(FATAL_ERROR
        "Oscillator backend contract FAILED (${_nv} issue(s)):\n${_report}\n"
        "See OSCILLATOR_BACKEND_CONTRACT_PLAN.md / spectral_oscillator_dispatch.h matrix.")
endif()

list(LENGTH _consumers _nc)
string(REPLACE ";" "\n    " _consumer_report "${_consumers}")
message(STATUS
    "Oscillator backend contract OK: Q15 v${_q15_version}, float v${_float_version}; "
    "${_nc} Q15 re-implementer(s) each carry a version-pin _Static_assert:\n    ${_consumer_report}")
