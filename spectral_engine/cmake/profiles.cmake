# profiles.cmake — SINGLE SOURCE OF TRUTH for build-profile compiler flags.
#
# This module defines the compiler-flag GROUPS (with rationale) that every build
# target's flag list is assembled from. host-config.cmake (desktop + host-embedded
# sim) and daisy-config.cmake (real Cortex-M7 firmware) CONSUME these groups; they
# decide platform discovery (SDKs, libraries, MCU arch) and which groups apply, but
# they do not invent flags. New flags/profiles are added HERE so there is one place
# to read "what does this build compile with, and why."
#
# Two build-profile PHILOSOPHIES (the contract — full table in
# docs/core_audit/reference/BUILD_PROFILES.md):
#
#   HOST  — desktop, the host-embedded simulation, and the test executables.
#           "Quality but well-optimized." -O3 + aggressive auto-vectorization + host
#           fast-math + native ISA + LTO, with full -Wall -Wextra kept on. AVX-512 is
#           capped OFF by default because a wider lane that down-clocks the core can
#           net-lose (see BUILD_PROFILES §AVX-512). Accuracy is preserved where the
#           engine's gated approximations require it; speed is taken where it is free.
#
#   FIRMWARE / EMBEDDED — the real STM32H7 Daisy build and its host-sim mirror.
#           "Aggressively minimal, fastest-for-the-target, deterministic." -O3 (or the
#           board opt level) + section garbage-collection + no unwind/exception tables
#           + MCU arch flags + SAFE fast-math, producing small, WCET-stable code.
#
# The math/precision decision is ONE knob per profile (not re-decided per flag):
#   SPECTRAL_REPRO_BUILD     (host + host-sim)  — ON drops the host/embedded fast-math
#                                                  group for a bit-reproducible build.
#   SPECTRAL_DAISY_SAFE_MATH (firmware)         — ON drops the firmware unsafe-math
#                                                  group for deterministic firmware.
# The conditional ASSEMBLY that reads those knobs lives in host-config/daisy-config
# (they run after options.cmake sets the knobs); this module only defines the groups.
#
# NOTE on CMAKE_BUILD_TYPE: it is intentionally NOT the optimization control here. The
# engine always compiles at its profile -O level (Debug adds -g / drops NDEBUG but
# stays optimized). See BUILD_PROFILES §CMAKE_BUILD_TYPE.

# Idempotent: this module is included by options.cmake (so it reaches BOTH the host
# configure and the separate Daisy-firmware sub-configure that consume the flag groups).
if(DEFINED SPECTRAL_PROFILES_INCLUDED)
    return()
endif()
set(SPECTRAL_PROFILES_INCLUDED TRUE)

# --- shared groups (used by more than one profile) ----------------------------------
set(SPECTRAL_FLAGS_O3                 -O3)
set(SPECTRAL_FLAGS_WALL_WEXTRA        -Wall -Wextra)
set(SPECTRAL_FLAGS_OMIT_FP            -fomit-frame-pointer)
# Section GC: emit each function/datum in its own section so the linker can dead-strip
# unreferenced code — size discipline for embedded; also used by the test executables.
set(SPECTRAL_FLAGS_SECTION_GC         -fdata-sections -ffunction-sections)

# --- HOST profile groups ------------------------------------------------------------
set(SPECTRAL_FLAGS_HOST_VECTORIZE     -funroll-loops -ftree-vectorize)
# Host fast-math: relaxed FP for speed (gated by NOT SPECTRAL_REPRO_BUILD).
set(SPECTRAL_FLAGS_HOST_FASTMATH      -ffast-math
                                      -fno-signed-zeros
                                      -fno-trapping-math
                                      -fassociative-math
                                      -freciprocal-math
                                      -ffp-contract=fast)
# Apple host extras (LTO + clang vectorizers + OpenMP-via-preprocessor). Library/OpenMP
# DISCOVERY stays in host-config; these are the static flags.
set(SPECTRAL_FLAGS_APPLE_HOST_EXTRA   -flto=thin -fvectorize -fslp-vectorize -Xpreprocessor -fopenmp)
# Linux/GCC host extras (LTO + polyhedral loop opts + interprocedural pointer analysis).
# ${OpenMP_C_FLAGS} is appended in host-config (it comes from find_package discovery).
set(SPECTRAL_FLAGS_LINUX_HOST_EXTRA   -flto=auto -floop-nest-optimize -fgraphite-identity -fipa-pta)

# Host native-ISA group. -march/-mtune=native tune for the build host. The x86-only
# SIMD-ISA flags are ARCH-GATED: -mavx2 enables 256-bit AVX2; -mno-avx512f caps AVX-512
# off (downclock). On arm64 those two are -Wunused-command-line-argument noise (NEON is
# already on via -march=native), so they are emitted only on x86. Provided as a function
# because the gate depends on CMAKE_SYSTEM_PROCESSOR.
function(spectral_profile_host_native_isa out_var)
    set(_isa -march=native -mtune=native)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
        list(APPEND _isa -mavx2 -mno-avx512f)
    endif()
    set(${out_var} "${_isa}" PARENT_SCOPE)
endfunction()

# --- EMBEDDED / FIRMWARE profile groups ---------------------------------------------
# Unsafe fast-math for the embedded paths (shared by the host-sim and firmware builds;
# gated OFF by SPECTRAL_REPRO_BUILD on the sim, by SPECTRAL_DAISY_SAFE_MATH on firmware).
set(SPECTRAL_FLAGS_EMBEDDED_FASTMATH  -ffast-math -funsafe-math-optimizations)
# Firmware-only minimalism: drop unwind tables (no C++ exceptions on the RT path).
set(SPECTRAL_FLAGS_FIRMWARE_NO_UNWIND -fno-unwind-tables -fno-asynchronous-unwind-tables)

# --- CUDA profile group -------------------------------------------------------------
set(SPECTRAL_FLAGS_CUDA               -O3
                                      --use_fast_math
                                      --fmad=true
                                      -Xcompiler=-fPIC
                                      -Xcompiler=-ffast-math
                                      -Xcompiler=-funroll-loops
                                      -Xcompiler=-fno-lto)

# --- shared helper: dead-strip link idiom for the test executables ------------------
# Previously copy-pasted across 13 test cmake files (kernel-hardening VI.4). Strips
# unreferenced code from an EXCLUDE_FROM_ALL test exe. Apple's linker -dead_strip needs
# only the link flag; the GNU/LLD path needs per-section emission to gc-sections. Matches
# the prior idiom flag-for-flag (section-GC compile flags on non-Apple only).
function(spectral_apply_dead_strip target_name)
    if(APPLE)
        target_link_options(${target_name} PRIVATE -Wl,-dead_strip)
    else()
        target_compile_options(${target_name} PRIVATE ${SPECTRAL_FLAGS_SECTION_GC})
        target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
    endif()
endfunction()
