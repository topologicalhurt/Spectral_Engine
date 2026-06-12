/* daisy_seed_sdram.h - the Daisy Seed memory-system constants, ONE place.
 *
 * Single source of truth for every hardware number the M7 performance model
 * (tools/spectral_tools/performance/embedded/memory_model.py) and the firmware
 * share: clock tree, FMC SDRAM timing, SDRAM geometry, cache geometry. The
 * Python side PARSES this header at runtime (C-truth rule, ADR-0002): it holds
 * no copy. Every value carries its calibration provenance next to it. Rename a
 * macro and the model fails loudly, not silently.
 *
 * TIMING POLICY (maintainer decision, 2026-06-12): the CHOSEN configuration is
 * the best-performance datasheet-derived set, not libDaisy's shipped one.
 * libDaisy's sdram.cpp ships debug-era timings (RPDelay=16, RCDDelay=10,
 * comments admit "started at 2") that are ~4x slower than the part needs, and
 * a SelfRefreshTime (tRAS) of 4 ticks = 40 ns, BELOW the 48 ns minimum its own
 * comment quotes. daisy_spectral_sdram_apply_timings() below re-programs
 * FMC_SDTR1 with the chosen set right after board init. Define
 * SPECTRAL_SDRAM_STOCK_TIMINGS to keep libDaisy's values (fallback if the
 * tightened timings ever misbehave on real hardware — they are derived from
 * datasheet minima, not yet verified on a board).
 *
 * CAS stays 3: lowering it requires reloading the SDRAM chip's mode register
 * (full FMC command sequence), for a 4-CPU-cycle linefill gain — not worth the
 * untestable risk until hardware exists. The TRP/TRCD fix is plain timing-
 * register writes and saves 88 CPU cycles per row miss.
 */
#ifndef DAISY_SEED_SDRAM_H
#define DAISY_SEED_SDRAM_H

#include <stdint.h>

/* --- Clock tree (libDaisy defaults; system.cpp FREQ_400MHZ path) ----------- */
/* CPU PLL1P: PLLN=200, HSE/PLLM=4, PLLP=2 -> 400 MHz [libDaisy system.cpp]   */
#define SPECTRAL_DAISY_CPU_HZ            400000000u
/* AXI/AHB = SYSCLK/2 (RCC_HCLK_DIV2) -> 200 MHz [libDaisy system.cpp;
 * AN4891 Rev1 p31: "bus matrices run at 200 MHz"]                            */
#define SPECTRAL_DAISY_AXI_HZ            200000000u
/* FMC kernel = PLL2R 200 MHz, SDCLK = ker/2 (FMC_SDRAM_CLOCK_PERIOD_2)
 * -> 100 MHz [libDaisy system.cpp + sdram.cpp]                               */
#define SPECTRAL_DAISY_SDCLK_HZ          100000000u

/* --- SDRAM geometry (AS4C16M32MSA-6BIN, 64 MB, on FMC bank 1) -------------- */
/* 32-bit data bus [libDaisy FMC_SDRAM_MEM_BUS_WIDTH_32]                      */
#define SPECTRAL_DAISY_SDRAM_BUS_BITS    32u
/* 4 internal banks [libDaisy FMC_SDRAM_INTERN_BANKS_NUM_4]                   */
#define SPECTRAL_DAISY_SDRAM_BANKS       4u
/* 512 columns x 32-bit = 2 KB per row per bank [AS4C16M32MSA organization]   */
#define SPECTRAL_DAISY_SDRAM_ROW_BYTES   2048u

/* --- Chosen FMC SDRAM timing, in SDCLK ticks @ 100 MHz (10 ns) -------------
 * Derivation: worst-of(Alliance AS4C32M16SA family datasheet "Common
 * Parameters" [tRCD 15ns, tRP 15ns, tRAS 45ns, tRC 65ns, tWR/tDPL 2 CLK],
 * libDaisy's quoted -6 part minima [tRAS 48ns, tRC 60ns]), each rounded UP.
 * The part-specific AS4C16M32MSA sheet is distribution-gated; the family
 * table + the in-tree quotes are the best obtainable provenance. Verify on
 * hardware before calling these final (no board as of pass 252).            */
#define SPECTRAL_DAISY_SDRAM_TMRD        2u   /* mode-register set, JEDEC 2 CLK   */
#define SPECTRAL_DAISY_SDRAM_TXSR        7u   /* self-refresh exit (libDaisy 7)   */
#define SPECTRAL_DAISY_SDRAM_TRAS        5u   /* 48 ns -> 5 (libDaisy ships 4!)   */
#define SPECTRAL_DAISY_SDRAM_TRC         7u   /* 65 ns -> 7                       */
#define SPECTRAL_DAISY_SDRAM_TWR         2u   /* tDPL 2 CLK                       */
#define SPECTRAL_DAISY_SDRAM_TRP         2u   /* 15 ns -> 2 (libDaisy ships 16)   */
#define SPECTRAL_DAISY_SDRAM_TRCD        2u   /* 15 ns -> 2 (libDaisy ships 10)   */
/* CAS latency in SDCLK ticks (controller AND chip mode register agree on 3). */
#define SPECTRAL_DAISY_SDRAM_CAS         3u

/* --- L1 cache geometry (STM32H7x3-class M7) -------------------------------- */
/* 16 KB data cache [AN4891 Rev1 p5 Table]; 4-way [TRM DDI 0489F 5.9];
 * 32 B lines [TRM 5.9] — must equal the engine-wide SPECTRAL_CACHE_LINE.     */
#define SPECTRAL_DAISY_DCACHE_BYTES      16384u
#define SPECTRAL_DAISY_DCACHE_WAYS       4u
#define SPECTRAL_DAISY_CACHE_LINE        32u

/* The engine-wide SPECTRAL_CACHE_LINE is conditional (32 on M7 targets, 64
 * portable), so the equality contract binds only where the M7 mapping
 * applies — a host TU including this header legitimately sees 64. */
#if defined(SPECTRAL_CACHE_LINE) && defined(SPECTRAL_ARM_M7) && SPECTRAL_ARM_M7
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(SPECTRAL_DAISY_CACHE_LINE == SPECTRAL_CACHE_LINE,
               "daisy cache-line constant diverged from the engine SSOT");
#endif
#endif

/* Re-program FMC_SDTR1 with the chosen timing set. Call once, after libDaisy
 * board init (hw.Init()) and before sustained SDRAM use; daisy_spectral_init()
 * calls it for you. Field layout per RM0433 (FMC_SDTR1): TMRD[3:0] TXSR[7:4]
 * TRAS[11:8] TRC[15:12] TWR[19:16] TRP[23:20] TRCD[27:24], encoded ticks-1.
 * Compiles to a no-op unless the STM32H7 device header is in the TU and stock
 * timings were not requested. */
static inline uint32_t daisy_spectral_sdram_apply_timings(void) {
#if defined(FMC_Bank5_6_R) && !defined(SPECTRAL_SDRAM_STOCK_TIMINGS)
    uint32_t sdtr =
        ((SPECTRAL_DAISY_SDRAM_TMRD - 1u) << 0)  |
        ((SPECTRAL_DAISY_SDRAM_TXSR - 1u) << 4)  |
        ((SPECTRAL_DAISY_SDRAM_TRAS - 1u) << 8)  |
        ((SPECTRAL_DAISY_SDRAM_TRC  - 1u) << 12) |
        ((SPECTRAL_DAISY_SDRAM_TWR  - 1u) << 16) |
        ((SPECTRAL_DAISY_SDRAM_TRP  - 1u) << 20) |
        ((SPECTRAL_DAISY_SDRAM_TRCD - 1u) << 24);
    __DSB();                              /* drain in-flight SDRAM traffic */
    FMC_Bank5_6_R->SDTR[0] = sdtr;
    __DSB();
    return sdtr;
#else
    return 0u;                            /* host build or stock opt-out */
#endif
}

#endif /* DAISY_SEED_SDRAM_H */
