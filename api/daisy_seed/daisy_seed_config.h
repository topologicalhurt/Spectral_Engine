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

#include "../../spectral_engine/spectral_config.h"
#include "../../spectral_engine/spectral_q15.h"

/* STM32H750 CPU */
#define DAISY_CPU_FREQ_HZ           480000000UL
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

/* SDRAM memory budget */
#define DAISY_SEGMENT_POOL_OFFSET   0x00000000UL
#define DAISY_SEGMENT_POOL_SIZE     (48UL * 1024 * 1024)
#define DAISY_AUDIO_POOL_OFFSET     0x03000000UL
#define DAISY_AUDIO_POOL_SIZE       (8UL * 1024 * 1024)
#define DAISY_WORK_POOL_OFFSET      0x03800000UL
#define DAISY_WORK_POOL_SIZE        (8UL * 1024 * 1024)

/* Segment limits */
#define DAISY_MAX_SEGMENTS          (DAISY_SEGMENT_POOL_SIZE / sizeof(SpectralSegmentQ15))
#define DAISY_MAX_SEGMENTS_SAFE     2000000
#define DAISY_MAX_ACTIVE            128

/* Parameter defaults and limits */
#define DAISY_DEFAULT_AMPLITUDE     0.8f
#define DAISY_STRETCH_MIN           0.25f
#define DAISY_STRETCH_MAX           4.0f
#define DAISY_Q214_UNITY            16384
#define DAISY_ADC_BITS              12
#define DAISY_ADC_MAX               4095
/* Q2.14: 0.25 = 4096, 4.0 = 65536, range = 61440 */
#define DAISY_STRETCH_Q214_MIN      4096
#define DAISY_STRETCH_Q214_RANGE    61440

/* Type aliases moved to daisy_seed_spectral.h (avoid circular dependency) */

/* Optimization */
#ifndef DAISY_USE_CMSIS_DSP
#define DAISY_USE_CMSIS_DSP         0
#endif
#define DAISY_OSC_LUT_BITS          SPECTRAL_OSC_LUT_BITS
#define DAISY_OSC_LUT_SIZE          SPECTRAL_OSC_LUT_SIZE
#define DAISY_OSC_LUT_MASK          SPECTRAL_OSC_LUT_MASK
/* Legacy aliases */
#define DAISY_SIN_LUT_BITS          DAISY_OSC_LUT_BITS
#define DAISY_SIN_LUT_SIZE          DAISY_OSC_LUT_SIZE
#define DAISY_SIN_LUT_MASK          DAISY_OSC_LUT_MASK
#define DAISY_SYNTH_UNROLL_FACTOR   4

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

/* Timing */
#define DAISY_AUDIO_PERIOD_US_48K   1000
#define DAISY_AUDIO_PERIOD_US_96K   500
#define DAISY_CYCLES_PER_BLOCK_48K  480000
#define DAISY_CYCLES_PER_BLOCK_96K  240000
#define DAISY_CYCLE_BUDGET_PERCENT  70
#define DAISY_CYCLES_BUDGET_48K     (DAISY_CYCLES_PER_BLOCK_48K * DAISY_CYCLE_BUDGET_PERCENT / 100)
#define DAISY_CYCLES_BUDGET_96K     (DAISY_CYCLES_PER_BLOCK_96K * DAISY_CYCLE_BUDGET_PERCENT / 100)

#ifndef DAISY_DEBUG
#define DAISY_DEBUG                 0
#endif

#endif /* DAISY_SEED_CONFIG_H */
