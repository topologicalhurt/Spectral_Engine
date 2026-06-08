/* spectral_mem.h - Device-agnostic memory-class placement.
 *
 * Core expresses memory placement by INTENT, never by device:
 *   SPECTRAL_MEM_FAST       hot data in the fastest / zero-wait memory (e.g. DTCM)
 *   SPECTRAL_MEM_FAST_CODE  hot code in the fastest instruction store  (e.g. ITCM)
 *   SPECTRAL_MEM_BULK       large / bulk data, possibly external/slower (e.g. SDRAM)
 *   SPECTRAL_CACHE_LINE     cache-line size in bytes (for alignment)
 *
 * Binding resolution (asm-generic style): a board support package overrides by
 * defining SPECTRAL_BSP_MEM_HEADER; otherwise a built-in default for the Cortex-M
 * reference target applies on a real Cortex-M cross-compile; otherwise the
 * portable no-op default (host, simulation).
 *
 * Phase E note: the built-in Cortex-M section names below are a default for the
 * current Daisy/STM32H7 reference target. They should migrate into the api/ BSP
 * (supplied via SPECTRAL_BSP_MEM_HEADER) once the embedded toolchain build is
 * exercised — core must carry no device names. Until then the default is gated to
 * a real Cortex-M cross-compile so host/sim builds never emit device sections.
 *
 * Included by spectral_config.h after SPECTRAL_ARM_M7 / SPECTRAL_EMBEDDED.
 */
#ifndef SPECTRAL_MEM_H
#define SPECTRAL_MEM_H

#include <stddef.h>

#if defined(SPECTRAL_BSP_MEM_HEADER)
#include SPECTRAL_BSP_MEM_HEADER   /* board-provided bindings; device lives in the BSP */
#endif

/* Built-in default for the Cortex-M reference target — only if no BSP supplied a
 * binding and we are on a real Cortex-M cross-compile. */
#if !defined(SPECTRAL_MEM_FAST) && SPECTRAL_ARM_M7 && SPECTRAL_EMBEDDED && \
    (defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__))
#define SPECTRAL_MEM_FAST       __attribute__((section(".dtcm_data")))
#define SPECTRAL_MEM_FAST_CODE  __attribute__((section(".itcm_text")))
#define SPECTRAL_MEM_BULK       __attribute__((section(".sdram_data")))
#endif

/* Portable no-op defaults (host, simulation, or any target without a binding). */
#ifndef SPECTRAL_MEM_FAST
#define SPECTRAL_MEM_FAST
#endif
#ifndef SPECTRAL_MEM_FAST_CODE
#define SPECTRAL_MEM_FAST_CODE
#endif
#ifndef SPECTRAL_MEM_BULK
#define SPECTRAL_MEM_BULK
#endif

/* Cache line: Cortex-M-class default 32, else portable 64 (a BSP may override). */
#if !defined(SPECTRAL_CACHE_LINE) && SPECTRAL_ARM_M7
#define SPECTRAL_CACHE_LINE     32
#endif
#ifndef SPECTRAL_CACHE_LINE
#define SPECTRAL_CACHE_LINE     64
#endif

#define SPECTRAL_CACHE_LINE_STRIDE (SPECTRAL_CACHE_LINE / sizeof(size_t))
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(SPECTRAL_CACHE_LINE % sizeof(size_t) == 0,
               "cache line must be multiple of size_t");
#endif

#endif /* SPECTRAL_MEM_H */
