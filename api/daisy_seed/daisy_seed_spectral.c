/* daisy_seed_spectral.c - Daisy Seed spectral synthesis wrapper */
#include "daisy_seed_spectral.h"
#include <string.h>
#include <math.h>

#ifdef DAISY_HAS_FATFS
#include "fatfs.h"
#include "ff.h"
#endif

/* Static memory pools */

static SpectralSegmentQ15 s_segment_pool[DAISY_MAX_SEGMENTS_SAFE] DAISY_SDRAM_BSS;
static q15_t s_osc_lut[SPECTRAL_OSC_LUT_SIZE + 1] DAISY_SRAM;
static uint8_t s_lut_initialized = 0;
static uint32_t s_memory_used = 0;

#ifdef DAISY_HAS_FATFS
static FATFS s_fatfs;
static uint8_t s_sd_mounted = 0;
#endif

/* Initialization */

DaisyResult daisy_spectral_init(DaisySpectralCtx* ctx, uint32_t sample_rate) {
    if (!ctx) return DAISY_ERR_PARAM;
    memset(ctx, 0, sizeof(DaisySpectralCtx));
    
    /* Validate sample rate */
    if (sample_rate != DAISY_SAMPLE_RATE && sample_rate != DAISY_SAMPLE_RATE_96K) {
        sample_rate = DAISY_SAMPLE_RATE;
    }
    
    /* Initialize oscillator LUT once */
    if (!s_lut_initialized) {
        spectral_lut_init_sine(s_osc_lut);
        s_lut_initialized = 1;
    }
    
    /* Initialize embedded synthesis context */
    spectral_arm32_init(&ctx->synth,
                           s_segment_pool,
                           DAISY_MAX_SEGMENTS_SAFE,
                           s_osc_lut,
                           sample_rate);
    
    /* Set Daisy defaults */
    spectral_arm32_set_amplitude(&ctx->synth, DAISY_DEFAULT_AMPLITUDE);
    
    /* Track memory usage */
    s_memory_used = sizeof(DaisySpectralCtx) + sizeof(s_osc_lut);
    ctx->total_memory_used = s_memory_used;
    
    return DAISY_OK;
}

void daisy_spectral_deinit(DaisySpectralCtx* ctx) {
    if (!ctx) return;
    s_memory_used = 0;
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
    
    /* Read directly into segment pool (segments is non-const in pool) */
    if (f_read(&file, (void*)ctx->synth.segments, seg_bytes, &bytes_read) != FR_OK || 
        bytes_read != seg_bytes) {
        f_close(&file);
        return DAISY_ERR_SD_READ;
    }
    
    f_close(&file);
    
    /* Update context state */
    ctx->synth.num_segments = header.num_segments;
    ctx->synth.output_length = header.output_length;
    ctx->total_memory_used = s_memory_used + seg_bytes;
    
    /* Reset playback position */
    spectral_arm32_reset(&ctx->synth);
    
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
    ctx->total_memory_used = s_memory_used + seg_bytes;
    
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
    if (segments_bytes) *segments_bytes = seg_bytes;
    if (total_bytes) *total_bytes = ctx->total_memory_used;
    if (available_bytes) *available_bytes = DAISY_SDRAM_SIZE - ctx->total_memory_used;
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
    return spectral_arm32_is_complete(&ctx->synth);
}

uint32_t daisy_spectral_get_position(const DaisySpectralCtx* ctx) {
    return spectral_arm32_get_position(&ctx->synth);
}

uint32_t daisy_spectral_get_duration(const DaisySpectralCtx* ctx) {
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

/* Compute expected data length for command */
static uint8_t uart_cmd_data_len(uint8_t cmd) {
    switch (cmd) {
        case DAISY_UART_CMD_PING:       return 0;
        case DAISY_UART_CMD_SET_STRETCH: return 4;  /* float */
        case DAISY_UART_CMD_SET_AMP:     return 4;  /* float */
        case DAISY_UART_CMD_RESET:       return 0;
        case DAISY_UART_CMD_SEEK:        return 4;  /* uint32 */
        case DAISY_UART_CMD_GET_STATUS:  return 0;
        case DAISY_UART_CMD_LOAD_FILE:   return 0;  /* Variable, null-terminated */
        default:                         return 0xFF; /* Unknown command */
    }
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
            DaisyResult result = daisy_spectral_load_sd(ctx, (const char*)s_uart_state.data);
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
        case UART_STATE_CMD:
            s_uart_state.cmd = byte;
            s_uart_state.expected_len = uart_cmd_data_len(byte);
            s_uart_state.data_len = 0;
            s_uart_state.checksum = byte;
            
            if (s_uart_state.expected_len == 0xFF) {
                /* Unknown command */
                s_uart_state.state = UART_STATE_IDLE;
                response_buf[0] = DAISY_UART_RESP_ERR;
                response_buf[1] = 0x00;
                response_buf[2] = response_buf[0] ^ response_buf[1];
                *response_len = 3;
                return 1;
            } else if (s_uart_state.expected_len == 0) {
                /* No data expected, wait for checksum */
                s_uart_state.state = UART_STATE_CHECKSUM;
            } else {
                s_uart_state.state = UART_STATE_DATA;
            }
            break;
            
        case UART_STATE_DATA:
            s_uart_state.data[s_uart_state.data_len++] = byte;
            s_uart_state.checksum ^= byte;
            
            /* For LOAD_FILE, look for null terminator */
            if (s_uart_state.cmd == DAISY_UART_CMD_LOAD_FILE) {
                if (byte == '\0') {
                    s_uart_state.state = UART_STATE_CHECKSUM;
                } else if (s_uart_state.data_len >= DAISY_UART_MAX_MSG_LEN - 1) {
                    /* Buffer overflow, reset */
                    s_uart_state.state = UART_STATE_IDLE;
                    response_buf[0] = DAISY_UART_RESP_ERR;
                    response_buf[1] = (uint8_t)(-DAISY_ERR_OVERFLOW);
                    response_buf[2] = response_buf[0] ^ response_buf[1];
                    *response_len = 3;
                    return 1;
                }
            } else if (s_uart_state.data_len >= s_uart_state.expected_len) {
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
