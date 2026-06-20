/* daisy_seed_spectral.c - Daisy Seed board API implementation */
#include "daisy_seed_spectral.h"
#include "daisy_seed_sdram.h"
#include "spectral_lut.h"
#include <string.h>
#include <math.h>

#ifdef DAISY_HAS_FATFS
#include "fatfs.h"
#include "ff.h"
#endif

/* Static memory pools.
 *
 * s_segment_pool is the SDRAM-resident segment store (DAISY_SDRAM_BSS -> .sdram_bss
 * @ 0xC0000000). Its L1 D-cache behaviour rides the MPU region attribute, which is
 * libDaisy's default and is NOT configured in this tree -- the load-bearing assumption
 * for sd-load coherency (see the f_read in daisy_spectral_load_sd). Normally write-back
 * write-allocate, which is why a DMA fill of the pool would need an invalidate.
 *
 * s_osc_lut is the 8 KB sine LUT, consumed ONLY by the generic (non-M7) gather oscillator;
 * the Daisy M7 build renders the gather-free coupled recurrence and never reads it during
 * synthesis (bandwidth-01: reclaimable once process()'s non-null precondition is gated off
 * the M7 path -- PERF-gated, maintainer-sequenced). */
static SpectralSegmentQ15 s_segment_pool[DAISY_MAX_SEGMENTS_SAFE] DAISY_SDRAM_BSS;
#if !SPECTRAL_ARM_M7   /* LUT oscillator (non-M7 builds); the M7 coupled osc reads no LUT (bandwidth-01) */
static q15_t s_osc_lut[SPECTRAL_OSC_LUT_SIZE + 1] DAISY_SRAM;
static uint8_t s_lut_initialized = 0;
#endif
static DaisySpectralCtx* s_active_ctx = NULL;

#ifdef DAISY_HAS_FATFS
static FATFS s_fatfs;
static uint8_t s_sd_mounted = 0;
#endif

static inline uint32_t daisy_base_memory_used(void) {
#if !SPECTRAL_ARM_M7
    return (uint32_t)(sizeof(DaisySpectralCtx) + sizeof(s_osc_lut));
#else
    return (uint32_t)(sizeof(DaisySpectralCtx));   /* no LUT on the coupled-osc M7 build */
#endif
}

/* Initialization */

DaisyResult daisy_spectral_init(DaisySpectralCtx* ctx, uint32_t sample_rate) {
    if (!ctx) return DAISY_ERR_PARAM;
    if (s_active_ctx && s_active_ctx != ctx) return DAISY_ERR_MEMORY;
    memset(ctx, 0, sizeof(DaisySpectralCtx));

    /* Replace libDaisy's debug-era FMC SDRAM timings with the chosen
     * datasheet-derived set (daisy_seed_sdram.h). No-op off-target or under
     * SPECTRAL_SDRAM_STOCK_TIMINGS; runs after the app's hw.Init(). */
    daisy_spectral_sdram_apply_timings();

    /* Validate sample rate */
    if (sample_rate != DAISY_SAMPLE_RATE && sample_rate != DAISY_SAMPLE_RATE_96K) {
        sample_rate = DAISY_SAMPLE_RATE;
    }
    
#if !SPECTRAL_ARM_M7
    /* Initialize oscillator LUT once (LUT-gather path; the M7 coupled osc never reads it) */
    if (!s_lut_initialized) {
        spectral_lut_init_sine(s_osc_lut);
        s_lut_initialized = 1;
    }
#endif

    /* Initialize embedded synthesis context */
    spectral_arm32_init(&ctx->synth,
                           s_segment_pool,
                           DAISY_MAX_SEGMENTS_SAFE,
#if !SPECTRAL_ARM_M7
                           s_osc_lut,
#else
                           (const q15_t*)0,   /* coupled osc: no LUT (bandwidth-01) */
#endif
                           sample_rate);
    
    /* Set Daisy defaults */
    spectral_arm32_set_amplitude(&ctx->synth, DAISY_DEFAULT_AMPLITUDE);
    
    /* Track memory usage */
    ctx->total_memory_used = daisy_base_memory_used();
    s_active_ctx = ctx;
    
    return DAISY_OK;
}

void daisy_spectral_deinit(DaisySpectralCtx* ctx) {
    if (!ctx) return;
    if (s_active_ctx == ctx) s_active_ctx = NULL;
    memset(ctx, 0, sizeof(DaisySpectralCtx));
}

/* SD Card */

#ifdef DAISY_HAS_FATFS

DaisyResult daisy_sd_init(void) {
    if (s_sd_mounted) return DAISY_OK;
    if (f_mount(&s_fatfs, "0:/", 1) != FR_OK) return DAISY_ERR_SD_MOUNT;
    s_sd_mounted = 1;
    return DAISY_OK;
}

void daisy_sd_deinit(void) {
    if (s_sd_mounted) {
        f_mount(NULL, "0:/", 0);
        s_sd_mounted = 0;
    }
}

int daisy_sd_is_mounted(void) { 
    return s_sd_mounted; 
}

DaisyResult daisy_spectral_load_sd(DaisySpectralCtx* ctx, const char* filename) {
    if (!ctx || !filename) return DAISY_ERR_PARAM;
    if (!s_sd_mounted && daisy_sd_init() != DAISY_OK) return DAISY_ERR_SD_MOUNT;
    
    FIL file;
    if (f_open(&file, filename, FA_READ) != FR_OK) return DAISY_ERR_SD_OPEN;
    
    SpqFileHeader header;
    UINT bytes_read;
    if (f_read(&file, &header, sizeof(header), &bytes_read) != FR_OK || 
        bytes_read != sizeof(header)) {
        f_close(&file);
        return DAISY_ERR_SD_READ;
    }
    
    if (header.magic != SPQ_FILE_MAGIC) {
        f_close(&file);
        return DAISY_ERR_FORMAT;
    }

    if (header.version != SPQ_FILE_VERSION) {
        f_close(&file);
        return DAISY_ERR_FORMAT;
    }
    
    if (header.num_segments > ctx->synth.segments_capacity) {
        f_close(&file);
        return DAISY_ERR_MEMORY;
    }
    
    uint32_t seg_bytes = header.num_segments * sizeof(SpectralSegmentQ15);
    
    /* Read directly into segment pool (segments is non-const in pool).
     *
     * CACHE-COHERENCY CONTRACT (sd-load): ctx->synth.segments is the SDRAM pool
     * (s_segment_pool, DAISY_SDRAM_BSS). If FatFS/SDMMC services this f_read by DMA
     * AND the MPU maps SDRAM write-back (libDaisy's default region attribute -- NOT
     * set in this tree; see s_segment_pool), the DMA write bypasses the L1 D-cache and
     * a later synthesis read can hit a STALE line. spectral_arm32_load_in_place() below
     * issues a dsb (write-completion ordering) but NOT a cache invalidate. The fix when
     * both conditions hold is a line-rounded SCB_InvalidateDCache_by_Addr over
     * [segments, seg_bytes) here -- the recipe already exists in
     * spectral_arm32_dma_rx_sync (spectral_cache_invalidate_range). Gated ON-TARGET:
     * confirm the f_read DMAs and the SDRAM attribute on hardware first (an
     * unconditional invalidate is wasted if f_read is CPU-copy or SDRAM non-cacheable). */
    if (f_read(&file, (void*)ctx->synth.segments, seg_bytes, &bytes_read) != FR_OK ||
        bytes_read != seg_bytes) {
        f_close(&file);
        return DAISY_ERR_SD_READ;
    }
    
    f_close(&file);

    /* The .spq file is an untrusted input. Validate the in-place-loaded pool
     * (per-segment bounds, monotonic ordering, simultaneous-active cap) and fence
     * the SDRAM writes before the payload can reach synthesis — a malformed file
     * must be rejected here, not overrun the fixed-size active[] array at render. */
    SpectralError verr = spectral_arm32_load_in_place(&ctx->synth, header.num_segments,
                                                      header.output_length);
    if (verr != SPECTRAL_OK) {
        return (verr == SPECTRAL_ERR_OVERFLOW || verr == SPECTRAL_ERR_MEMORY)
                   ? DAISY_ERR_MEMORY : DAISY_ERR_FORMAT;
    }
    ctx->total_memory_used = daisy_base_memory_used() + seg_bytes;

    return DAISY_OK;
}

#else /* !DAISY_HAS_FATFS */

DaisyResult daisy_sd_init(void) { return DAISY_ERR_IO; }
void daisy_sd_deinit(void) {}
int daisy_sd_is_mounted(void) { return 0; }
DaisyResult daisy_spectral_load_sd(DaisySpectralCtx* ctx, const char* filename) {
    (void)ctx; (void)filename;
    return DAISY_ERR_IO;
}

#endif /* DAISY_HAS_FATFS */

/* Buffer Loading */

DaisyResult daisy_spectral_load_buffer(DaisySpectralCtx* ctx,
                                       const DaisySegmentQ15* data,
                                       uint32_t num_segments,
                                       uint32_t output_len) {
    if (!ctx || !data) return DAISY_ERR_PARAM;
    
    /* Delegate to embedded loader */
    SpectralError result = spectral_arm32_load(&ctx->synth, data, num_segments, output_len);
    if (result != SPECTRAL_OK) {
        if (result == SPECTRAL_ERR_OVERFLOW || result == SPECTRAL_ERR_MEMORY) {
            return DAISY_ERR_MEMORY;
        }
        return DAISY_ERR_PARAM;
    }
    
    /* Update memory tracking */
    uint32_t seg_bytes = num_segments * sizeof(SpectralSegmentQ15);
    ctx->total_memory_used = daisy_base_memory_used() + seg_bytes;
    
    return DAISY_OK;
}

void daisy_spectral_get_memory(const DaisySpectralCtx* ctx,
                               uint32_t* segments_bytes,
                               uint32_t* total_bytes,
                               uint32_t* available_bytes) {
    if (!ctx) {
        if (segments_bytes) *segments_bytes = 0;
        if (total_bytes) *total_bytes = 0;
        if (available_bytes) *available_bytes = DAISY_SDRAM_SIZE;
        return;
    }
    
    uint32_t seg_bytes = ctx->synth.num_segments * sizeof(SpectralSegmentQ15);
    uint32_t total_used = ctx->total_memory_used;
    if (total_used > DAISY_SDRAM_SIZE) total_used = DAISY_SDRAM_SIZE;
    if (segments_bytes) *segments_bytes = seg_bytes;
    if (total_bytes) *total_bytes = total_used;
    if (available_bytes) *available_bytes = DAISY_SDRAM_SIZE - total_used;
}

/* Parameters */

void daisy_spectral_set_stretch(DaisySpectralCtx* ctx, float stretch) {
    if (!ctx) return;
    stretch = CLAMP(stretch, DAISY_STRETCH_MIN, DAISY_STRETCH_MAX);
    spectral_arm32_set_stretch(&ctx->synth, stretch);
}

void daisy_spectral_set_amplitude(DaisySpectralCtx* ctx, float amplitude) {
    if (!ctx) return;
    amplitude = CLAMP(amplitude, 0.0f, 1.0f);
    spectral_arm32_set_amplitude(&ctx->synth, amplitude);
}

void daisy_spectral_set_params_adc(DaisySpectralCtx* ctx,
                                   uint16_t stretch_adc,
                                   uint16_t amplitude_adc) {
    if (!ctx) return;
    
    /* Convert 12-bit ADC to stretch (0.25 - 4.0) */
    float stretch = DAISY_STRETCH_MIN + 
        ((float)stretch_adc / DAISY_ADC_MAX) * (DAISY_STRETCH_MAX - DAISY_STRETCH_MIN);
    spectral_arm32_set_stretch(&ctx->synth, stretch);
    
    /* Convert 12-bit ADC to amplitude (0.0 - 1.0) */
    float amplitude = (float)amplitude_adc / DAISY_ADC_MAX;
    spectral_arm32_set_amplitude(&ctx->synth, amplitude);
}

/* Playback Control */

void daisy_spectral_reset(DaisySpectralCtx* ctx) {
    if (!ctx) return;
    spectral_arm32_reset(&ctx->synth);
}

void daisy_spectral_seek(DaisySpectralCtx* ctx, uint32_t sample_pos) {
    if (!ctx) return;
    spectral_arm32_seek(&ctx->synth, sample_pos);
}

int daisy_spectral_is_complete(const DaisySpectralCtx* ctx) {
    if (!ctx) return 1;
    return spectral_arm32_is_complete(&ctx->synth);
}

uint32_t daisy_spectral_get_position(const DaisySpectralCtx* ctx) {
    if (!ctx) return 0;
    return spectral_arm32_get_position(&ctx->synth);
}

uint32_t daisy_spectral_get_duration(const DaisySpectralCtx* ctx) {
    if (!ctx) return 0;
    return spectral_arm32_get_duration(&ctx->synth);
}

/* Audio Processing */

DAISY_HOT DAISY_FLATTEN
uint32_t daisy_spectral_process_q15(DaisySpectralCtx* ctx,
                                    q15_t* out_left,
                                    q15_t* out_right,
                                    uint32_t num_samples) {
    if (!ctx) return 0;
    return spectral_arm32_process(&ctx->synth, out_left, out_right, num_samples);
}

DAISY_HOT
uint32_t daisy_spectral_process_interleaved(DaisySpectralCtx* ctx,
                                            q15_t* out_interleaved,
                                            uint32_t num_samples) {
    if (!ctx) return 0;
    return spectral_arm32_process_interleaved(&ctx->synth, out_interleaved, num_samples);
}

/*
 * UART Protocol Implementation
 * 
 * State machine for parsing incoming UART bytes and executing commands.
 */

typedef enum {
    UART_STATE_IDLE,
    UART_STATE_CMD,
    UART_STATE_DATA,
    UART_STATE_CHECKSUM
} UartState;

static struct {
    UartState state;
    uint8_t cmd;
    uint8_t data[DAISY_UART_MAX_MSG_LEN];
    uint8_t data_len;
    uint8_t expected_len;
    uint8_t checksum;
} s_uart_state = { .state = UART_STATE_IDLE };

#define UART_CMD_LEN_UNKNOWN  0xFFu
#define UART_CMD_LEN_VARIABLE 0xFEu

/* Compute expected data length for command */
static uint8_t uart_cmd_data_len(uint8_t cmd) {
    switch (cmd) {
        case DAISY_UART_CMD_PING:       return 0;
        case DAISY_UART_CMD_SET_STRETCH: return 4;  /* float */
        case DAISY_UART_CMD_SET_AMP:     return 4;  /* float */
        case DAISY_UART_CMD_RESET:       return 0;
        case DAISY_UART_CMD_SEEK:        return 4;  /* uint32 */
        case DAISY_UART_CMD_GET_STATUS:  return 0;
        case DAISY_UART_CMD_LOAD_FILE:   return UART_CMD_LEN_VARIABLE;  /* Null-terminated filename */
        default:                         return UART_CMD_LEN_UNKNOWN; /* Unknown command */
    }
}

static int uart_begin_command(uint8_t cmd) {
    s_uart_state.cmd = cmd;
    s_uart_state.expected_len = uart_cmd_data_len(cmd);
    s_uart_state.data_len = 0;
    s_uart_state.checksum = cmd;

    if (s_uart_state.expected_len == UART_CMD_LEN_UNKNOWN) {
        s_uart_state.state = UART_STATE_IDLE;
        return 0;
    }

    s_uart_state.state = (s_uart_state.expected_len == 0) ? UART_STATE_CHECKSUM : UART_STATE_DATA;
    return 1;
}

/* Execute command and build response */
static void uart_execute(DaisySpectralCtx* ctx, uint8_t* response_buf, uint8_t* response_len) {
    uint8_t checksum = 0;
    
    switch (s_uart_state.cmd) {
        case DAISY_UART_CMD_PING:
            response_buf[0] = DAISY_UART_RESP_PONG;
            response_buf[1] = DAISY_UART_RESP_PONG;  /* Checksum */
            *response_len = 2;
            return;
            
        case DAISY_UART_CMD_SET_STRETCH: {
            float stretch;
            memcpy(&stretch, s_uart_state.data, sizeof(float));
            daisy_spectral_set_stretch(ctx, stretch);
            response_buf[0] = DAISY_UART_RESP_OK;
            response_buf[1] = DAISY_UART_RESP_OK;
            *response_len = 2;
            return;
        }
        
        case DAISY_UART_CMD_SET_AMP: {
            float amp;
            memcpy(&amp, s_uart_state.data, sizeof(float));
            daisy_spectral_set_amplitude(ctx, amp);
            response_buf[0] = DAISY_UART_RESP_OK;
            response_buf[1] = DAISY_UART_RESP_OK;
            *response_len = 2;
            return;
        }
        
        case DAISY_UART_CMD_RESET:
            daisy_spectral_reset(ctx);
            response_buf[0] = DAISY_UART_RESP_OK;
            response_buf[1] = DAISY_UART_RESP_OK;
            *response_len = 2;
            return;
            
        case DAISY_UART_CMD_SEEK: {
            uint32_t pos;
            memcpy(&pos, s_uart_state.data, sizeof(uint32_t));
            daisy_spectral_seek(ctx, pos);
            response_buf[0] = DAISY_UART_RESP_OK;
            response_buf[1] = DAISY_UART_RESP_OK;
            *response_len = 2;
            return;
        }
        
        case DAISY_UART_CMD_GET_STATUS: {
            DaisyStatusResponse status = {
                .position = daisy_spectral_get_position(ctx),
                .duration = daisy_spectral_get_duration(ctx),
                .is_complete = (uint8_t)daisy_spectral_is_complete(ctx),
                .is_loaded = (ctx && ctx->synth.num_segments > 0) ? 1 : 0,
                .reserved = 0
            };
            response_buf[0] = DAISY_UART_RESP_STATUS;
            memcpy(&response_buf[1], &status, sizeof(status));
            /* Compute checksum */
            checksum = response_buf[0];
            for (uint8_t i = 1; i < 1 + sizeof(status); i++) {
                checksum ^= response_buf[i];
            }
            response_buf[1 + sizeof(status)] = checksum;
            *response_len = 2 + sizeof(status);
            return;
        }
        
        case DAISY_UART_CMD_LOAD_FILE: {
#ifdef DAISY_HAS_FATFS
            /* Reject path traversal: disallow ".." components */
            const char* fname = (const char*)s_uart_state.data;
            int path_ok = (fname[0] != '\0');
            for (const char* p = fname; *p && path_ok; p++) {
                if (p[0] == '.' && p[1] == '.' &&
                    (p[2] == '/' || p[2] == '\\' || p[2] == '\0')) {
                    path_ok = 0;
                }
            }
            if (!path_ok) {
                response_buf[0] = DAISY_UART_RESP_ERR;
                response_buf[1] = (uint8_t)(-DAISY_ERR_PARAM);
                response_buf[2] = response_buf[0] ^ response_buf[1];
                *response_len = 3;
                return;
            }
            DaisyResult result = daisy_spectral_load_sd(ctx, fname);
            if (result == DAISY_OK) {
                response_buf[0] = DAISY_UART_RESP_OK;
                response_buf[1] = DAISY_UART_RESP_OK;
                *response_len = 2;
            } else {
                response_buf[0] = DAISY_UART_RESP_ERR;
                response_buf[1] = (uint8_t)(-result);  /* Error code as positive value */
                response_buf[2] = response_buf[0] ^ response_buf[1];
                *response_len = 3;
            }
#else
            response_buf[0] = DAISY_UART_RESP_ERR;
            response_buf[1] = (uint8_t)(-DAISY_ERR_IO);
            response_buf[2] = response_buf[0] ^ response_buf[1];
            *response_len = 3;
#endif
            return;
        }
        
        default:
            response_buf[0] = DAISY_UART_RESP_ERR;
            response_buf[1] = 0x00;  /* Unknown command */
            response_buf[2] = response_buf[0] ^ response_buf[1];
            *response_len = 3;
            return;
    }
}

int daisy_uart_process_byte(DaisySpectralCtx* ctx, 
                            uint8_t byte,
                            uint8_t* response_buf,
                            uint8_t* response_len) {
    *response_len = 0;
    
    switch (s_uart_state.state) {
        case UART_STATE_IDLE:
            if (byte == DAISY_UART_SYNC) {
                s_uart_state.state = UART_STATE_CMD;
                break;
            }
            if (!uart_begin_command(byte)) break; /* Ignore framing noise in idle state. */
            break;

        case UART_STATE_CMD:
            if (!uart_begin_command(byte)) {
                response_buf[0] = DAISY_UART_RESP_ERR;
                response_buf[1] = 0x00;
                response_buf[2] = response_buf[0] ^ response_buf[1];
                *response_len = 3;
                return 1;
            }
            break;
            
        case UART_STATE_DATA:
            if (s_uart_state.data_len >= DAISY_UART_MAX_MSG_LEN) {
                s_uart_state.state = UART_STATE_IDLE;
                response_buf[0] = DAISY_UART_RESP_ERR;
                response_buf[1] = (uint8_t)(-DAISY_ERR_OVERFLOW);
                response_buf[2] = response_buf[0] ^ response_buf[1];
                *response_len = 3;
                return 1;
            }
            s_uart_state.data[s_uart_state.data_len++] = byte;
            s_uart_state.checksum ^= byte;
            
            /* For LOAD_FILE, look for null terminator */
            if (s_uart_state.cmd == DAISY_UART_CMD_LOAD_FILE) {
                if (byte == '\0') {
                    s_uart_state.state = UART_STATE_CHECKSUM;
                } else if (s_uart_state.data_len >= DAISY_UART_MAX_MSG_LEN) {
                    /* Buffer overflow, reset */
                    s_uart_state.state = UART_STATE_IDLE;
                    response_buf[0] = DAISY_UART_RESP_ERR;
                    response_buf[1] = (uint8_t)(-DAISY_ERR_OVERFLOW);
                    response_buf[2] = response_buf[0] ^ response_buf[1];
                    *response_len = 3;
                    return 1;
                }
            } else if (s_uart_state.expected_len != UART_CMD_LEN_VARIABLE &&
                       s_uart_state.data_len >= s_uart_state.expected_len) {
                s_uart_state.state = UART_STATE_CHECKSUM;
            }
            break;
            
        case UART_STATE_CHECKSUM:
            s_uart_state.state = UART_STATE_IDLE;
            
            if (byte == s_uart_state.checksum) {
                /* Checksum OK, execute command */
                uart_execute(ctx, response_buf, response_len);
            } else {
                /* Checksum error */
                response_buf[0] = DAISY_UART_RESP_ERR;
                response_buf[1] = 0xFE;  /* Checksum error code */
                response_buf[2] = response_buf[0] ^ response_buf[1];
                *response_len = 3;
            }
            return 1;
    }
    
    return 0;
}

void daisy_uart_reset(void) {
    s_uart_state.state = UART_STATE_IDLE;
    s_uart_state.data_len = 0;
    s_uart_state.checksum = 0;
}
