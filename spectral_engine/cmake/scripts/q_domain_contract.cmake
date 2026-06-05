# Q-domain contract checker (run as: cmake -DSPECTRAL_Q_CONTRACT_ROOT=<dir> -P this).
#
# Pins the "no float/Q mixing outside boundary macros" discipline from
# QTYPE_DOMAIN_PLAN.md Q0, as a source scan (no compilation). Two rules:
#
#   RULE 1 — boundary-macro confinement. The raw Q-scale conversion constants
#   (SPECTRAL_Q15_SCALE / SPECTRAL_INV_Q15_SCALE / SPECTRAL_Q31_SCALE /
#   SPECTRAL_INV_Q31_SCALE) may appear ONLY in the sanctioned boundary-macro
#   definition sites (the allowlist below). Anywhere else means an ad-hoc
#   float<->Q conversion that bypassed the named macros (FLOAT_TO_Q15 /
#   Q15_TO_FLOAT / FLOAT_TO_Q31 / Q31_TO_FLOAT / SPECTRAL_SAMPLE_TO_FLOAT /
#   FLOAT_TO_SPECTRAL_SAMPLE) — the "float kernel does raw Q math" failure.
#
#   RULE 2 — Q-domain region purity. Code bracketed by comment-led markers
#       // SPECTRAL_Q_DOMAIN BEGIN ... // SPECTRAL_Q_DOMAIN END
#   is pure fixed point: no `float`/`double` token may appear inside — the
#   "Q kernel does float arithmetic" failure. Markers must balance.
#
# Implementation note: lines are walked with string(FIND/SUBSTRING) on a quoted
# string, never `foreach(x ${content})`. C source contains `;`, `[`, `]`, and `\`
# which CMake's list re-parse would mangle (a `[` opens a bracket argument; `;`
# splits elements; blank lines get dropped, corrupting line numbers). The manual
# walk sidesteps all of that.
#
# A FATAL_ERROR (nonzero exit) fails the q_domain_contract CTest.

if(NOT DEFINED SPECTRAL_Q_CONTRACT_ROOT)
    message(FATAL_ERROR "q_domain_contract: SPECTRAL_Q_CONTRACT_ROOT not set")
endif()

# Files permitted to reference the raw scale constants (matched by basename):
#   spectral_consts.h  — defines the constants
#   spectral_q15.h     — FLOAT_TO_Q15 / Q15_TO_FLOAT / FLOAT_TO_Q31 / Q31_TO_FLOAT
#   spectral_config.h  — SPECTRAL_SAMPLE_TO_FLOAT / FLOAT_TO_SPECTRAL_SAMPLE
set(_allow
    "spectral_consts.h"
    "spectral_q15.h"
    "spectral_config.h")

set(_scale_consts
    "SPECTRAL_INV_Q15_SCALE"
    "SPECTRAL_Q15_SCALE"
    "SPECTRAL_INV_Q31_SCALE"
    "SPECTRAL_Q31_SCALE")

file(GLOB_RECURSE _sources
    "${SPECTRAL_Q_CONTRACT_ROOT}/*.c"
    "${SPECTRAL_Q_CONTRACT_ROOT}/*.h")

set(_violations "")
set(_files_scanned 0)
set(_regions_scanned 0)

foreach(_file IN LISTS _sources)
    get_filename_component(_base "${_file}" NAME)
    math(EXPR _files_scanned "${_files_scanned} + 1")

    file(READ "${_file}" _content)
    string(REPLACE "\r" "" _content "${_content}")

    list(FIND _allow "${_base}" _allow_idx)
    if(_allow_idx EQUAL -1)
        set(_check_scale TRUE)
    else()
        set(_check_scale FALSE)
    endif()

    # Fast path: only files that actually mention a relevant token need the
    # per-line walk. Everything else (the vast majority) is skipped.
    set(_has_marker FALSE)
    string(FIND "${_content}" "SPECTRAL_Q_DOMAIN" _mk)
    if(NOT _mk EQUAL -1)
        set(_has_marker TRUE)
    endif()
    set(_has_scale FALSE)
    if(_check_scale)
        foreach(_c IN LISTS _scale_consts)
            string(FIND "${_content}" "${_c}" _sp)
            if(NOT _sp EQUAL -1)
                set(_has_scale TRUE)
                break()
            endif()
        endforeach()
    endif()
    if(NOT _has_marker AND NOT _has_scale)
        continue()
    endif()

    set(_lineno 0)
    set(_in_region FALSE)
    string(LENGTH "${_content}" _clen)
    while(_clen GREATER 0)
        math(EXPR _lineno "${_lineno} + 1")
        string(FIND "${_content}" "\n" _nl)
        if(_nl EQUAL -1)
            set(_line "${_content}")
            set(_content "")
        else()
            string(SUBSTRING "${_content}" 0 ${_nl} _line)
            math(EXPR _rest "${_nl} + 1")
            string(SUBSTRING "${_content}" ${_rest} -1 _content)
        endif()
        string(LENGTH "${_content}" _clen)

        # RULE 1 — raw scale constants outside the boundary allowlist.
        if(_check_scale AND _has_scale)
            foreach(_c IN LISTS _scale_consts)
                string(FIND "${_line}" "${_c}" _pos)
                if(NOT _pos EQUAL -1)
                    list(APPEND _violations
                        "  ${_file}:${_lineno}: raw '${_c}' outside a boundary macro (use FLOAT_TO_Q15/Q15_TO_FLOAT/SPECTRAL_SAMPLE_TO_FLOAT/...)")
                endif()
            endforeach()
        endif()

        # RULE 2 — region markers + float/double purity inside a region.
        # A real marker is a comment-led line that STARTS with the token (after
        # indentation), e.g. `// SPECTRAL_Q_DOMAIN BEGIN` or `/* ... */`. Anchored
        # so prose that merely mentions the markers is not mistaken for one.
        string(STRIP "${_line}" _stripped)
        string(REGEX MATCH "^(//|/\\*)[ \t]*SPECTRAL_Q_DOMAIN BEGIN" _begin_m "${_stripped}")
        string(REGEX MATCH "^(//|/\\*)[ \t]*SPECTRAL_Q_DOMAIN END" _end_m "${_stripped}")
        if(NOT _begin_m STREQUAL "")
            if(_in_region)
                list(APPEND _violations
                    "  ${_file}:${_lineno}: nested SPECTRAL_Q_DOMAIN BEGIN (region already open)")
            endif()
            set(_in_region TRUE)
            math(EXPR _regions_scanned "${_regions_scanned} + 1")
        elseif(NOT _end_m STREQUAL "")
            if(NOT _in_region)
                list(APPEND _violations
                    "  ${_file}:${_lineno}: SPECTRAL_Q_DOMAIN END without an open region")
            endif()
            set(_in_region FALSE)
        elseif(_in_region)
            string(REGEX MATCH "(^|[^A-Za-z0-9_])(float|double)([^A-Za-z0-9_]|$)" _fm "${_line}")
            if(NOT _fm STREQUAL "")
                list(APPEND _violations
                    "  ${_file}:${_lineno}: float/double inside a SPECTRAL_Q_DOMAIN region (pure fixed point only)")
            endif()
        endif()
    endwhile()

    if(_in_region)
        list(APPEND _violations
            "  ${_file}: SPECTRAL_Q_DOMAIN BEGIN without a matching END")
    endif()
endforeach()

list(LENGTH _violations _nv)
if(_nv GREATER 0)
    string(REPLACE ";" "\n" _report "${_violations}")
    message(FATAL_ERROR
        "Q-domain contract FAILED (${_nv} violation(s)):\n${_report}\n"
        "See QTYPE_DOMAIN_PLAN.md / spectral_q15.h for the boundary-macro rule.")
endif()

message(STATUS
    "Q-domain contract OK: ${_files_scanned} files scanned, "
    "${_regions_scanned} Q-domain region(s) verified pure, "
    "scale constants confined to boundary macros.")
