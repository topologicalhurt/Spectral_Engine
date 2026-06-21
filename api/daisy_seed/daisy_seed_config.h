/* daisy_seed_config.h - Daisy Seed Platform Configuration */
#ifndef DAISY_SEED_CONFIG_H
#define DAISY_SEED_CONFIG_H

/* Platform identifier - before spectral_config.h */
#define SPECTRAL_PLATFORM_DAISY     1

/* Enable embedded mode */
#undef  SPECTRAL_EMBEDDED
#define SPECTRAL_EMBEDDED           1
#undef  SPECTRAL_COMPACT_SEG
#define SPECTRAL_COMPACT_SEG        1
#undef  SPECTRAL_NO_PERF
#define SPECTRAL_NO_PERF            1

#include "../../spectral_engine/core/spectral_config.h"
#include "../../spectral_engine/core/spectral_q.h"

/* STM32H750 CPU. The core clock lives in daisy_seed_sdram.h
 * (SPECTRAL_DAISY_CPU_HZ), which the perf model and WCET parse. */
#define DAISY_HAS_FPU               1
#define DAISY_HAS_DSP               1

/* Memory map */
#define DAISY_SDRAM_BASE            0xC0000000UL
#define DAISY_SDRAM_SIZE            (64UL * 1024 * 1024)
#define DAISY_QSPI_BASE             0x90000000UL
#define DAISY_QSPI_SIZE             (8UL * 1024 * 1024)
#define DAISY_AXI_SRAM_BASE         0x24000000UL
#define DAISY_AXI_SRAM_SIZE         (512UL * 1024)
#define DAISY_SRAM1_BASE            0x30000000UL
#define DAISY_SRAM1_SIZE            (128UL * 1024)
#define DAISY_DTCM_BASE             0x20000000UL
#define DAISY_DTCM_SIZE             (128UL * 1024)

/* Audio (WM8731 codec) */
#define DAISY_SAMPLE_RATE           48000
#define DAISY_SAMPLE_RATE_96K       96000
#define DAISY_BIT_DEPTH             24
#define DAISY_AUDIO_CHANNELS        2
#define DAISY_AUDIO_BLOCK_SIZE      48

/* SDRAM memory budget: three pools tile the 64 MB SDRAM exactly
 * (48 segment + 8 audio + 8 work); each offset is the previous pool's end.
 * The tiling is enforced below, so resizing one pool without rebalancing
 * the others fails the build instead of silently overlapping. */
#define DAISY_SEGMENT_POOL_OFFSET   0x00000000UL
#define DAISY_SEGMENT_POOL_SIZE     (48UL * 1024 * 1024)
#define DAISY_AUDIO_POOL_OFFSET     0x03000000UL
#define DAISY_AUDIO_POOL_SIZE       (8UL * 1024 * 1024)
#define DAISY_WORK_POOL_OFFSET      0x03800000UL
#define DAISY_WORK_POOL_SIZE        (8UL * 1024 * 1024)

_Static_assert(DAISY_SEGMENT_POOL_OFFSET + DAISY_SEGMENT_POOL_SIZE
                   == DAISY_AUDIO_POOL_OFFSET,
               "audio pool must start where the segment pool ends");
_Static_assert(DAISY_AUDIO_POOL_OFFSET + DAISY_AUDIO_POOL_SIZE
                   == DAISY_WORK_POOL_OFFSET,
               "work pool must start where the audio pool ends");
_Static_assert(DAISY_WORK_POOL_OFFSET + DAISY_WORK_POOL_SIZE
                   == DAISY_SDRAM_SIZE,
               "pools must tile the SDRAM exactly");

/* Segment limits */
#define DAISY_MAX_SEGMENTS          (DAISY_SEGMENT_POOL_SIZE / sizeof(SpectralSegmentQ15))
/* [chosen: round figure under the ~3.1M (48 MB / 16 B) pool capacity,
 * leaving slack for pool bookkeeping.] */
#define DAISY_MAX_SEGMENTS_SAFE     2000000
/* Conservative legacy reference for the active-voice budget. NOT the admission cap: admission is
 * gated on SPECTRAL_ARM32_ACTIVE_CAP (spectral_config.h), which defaults to the storage bound
 * SPECTRAL_ARM32_MAX_ACTIVE (512). 128 is the LEGACY figure and is intentionally not wired to the
 * cap -- the measured per-voice WCET ceiling is far higher: the oscillator bank reaches ≈520 voices
 * @400MHz / ~625 @480MHz at the real 48-sample codec block with the 16 cyc/voice-sample kernel, so
 * the 512 storage default is itself inside budget (and the inverse-FFT path is no longer required
 * for the 512-voice real-time target). To tighten the real-time cap below storage, set
 * SPECTRAL_ARM32_ACTIVE_CAP from a fresh WCET derivation -- not by hand-wiring this 128. */
#define DAISY_MAX_ACTIVE            128

/* Parameter defaults and limits */
#define DAISY_DEFAULT_AMPLITUDE     0.8f
#define DAISY_STRETCH_MIN           0.25f
#define DAISY_STRETCH_MAX           4.0f
#define DAISY_ADC_MAX               4095

/* Type aliases moved to daisy_seed_spectral.h (avoid circular dependency) */

/* Optimization */
#ifndef DAISY_USE_CMSIS_DSP
#define DAISY_USE_CMSIS_DSP         0
#endif

/* GCC section attributes - only valid on ARM embedded targets */
#if defined(__GNUC__) && defined(__arm__)
    #define DAISY_SDRAM_BSS         __attribute__((section(".sdram_bss")))
    #define DAISY_DTCMRAM           __attribute__((section(".dtcmram_bss")))
    #define DAISY_SRAM              __attribute__((section(".sram_bss")))
    #define DAISY_ALIGN(n)          __attribute__((aligned(n)))
    #define DAISY_PACKED            __attribute__((packed))
    #define DAISY_HOT               __attribute__((hot))
    #define DAISY_FLATTEN           __attribute__((flatten))
#elif defined(__GNUC__)
    /* Desktop compilation (e.g., for syntax checking) */
    #define DAISY_SDRAM_BSS
    #define DAISY_DTCMRAM
    #define DAISY_SRAM
    #define DAISY_ALIGN(n)          __attribute__((aligned(n)))
    #define DAISY_PACKED            __attribute__((packed))
    #define DAISY_HOT
    #define DAISY_FLATTEN
#else
    #define DAISY_SDRAM_BSS
    #define DAISY_DTCMRAM
    #define DAISY_SRAM
    #define DAISY_ALIGN(n)
    #define DAISY_PACKED
    #define DAISY_HOT
    #define DAISY_FLATTEN
#endif

/* GPIO pins */
#define DAISY_PIN_STRETCH_POT       15
#define DAISY_PIN_VOLUME_POT        16
#define DAISY_PIN_PLAY_BTN          28
#define DAISY_PIN_RESET_BTN         27
#define DAISY_PIN_LED_STATUS        25
#define DAISY_LED_ONBOARD           0

#ifndef DAISY_DEBUG
#define DAISY_DEBUG                 0
#endif

#endif /* DAISY_SEED_CONFIG_H */
