/* daisy_seed_spectral.h - Daisy Seed Spectral Engine API */
#ifndef DAISY_SEED_SPECTRAL_H
#define DAISY_SEED_SPECTRAL_H

#include "daisy_seed_config.h"
#include "../../spectral_engine/spectral_synth_embedded.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DAISY_OK              =  0,
    DAISY_ERR_MEMORY      = -1,
    DAISY_ERR_PARAM       = -2,
    DAISY_ERR_FORMAT      = -3,
    DAISY_ERR_IO          = -4,
    DAISY_ERR_SD_MOUNT    = -5,
    DAISY_ERR_SD_OPEN     = -6,
    DAISY_ERR_SD_READ     = -7,
    DAISY_ERR_OVERFLOW    = -8,
    DAISY_ERR_NOT_INIT    = -9
} DaisyResult;

typedef struct {
    SpectralArm32Ctx synth;
    uint32_t total_memory_used;
} DaisySpectralCtx;

typedef SpectralActiveSegment DaisyActiveSegQ15;
typedef SpectralSegmentQ15 DaisySegmentQ15;

/* Initialization */
DaisyResult daisy_spectral_init(DaisySpectralCtx* ctx, uint32_t sample_rate);
void daisy_spectral_deinit(DaisySpectralCtx* ctx);

/* SD card */
DaisyResult daisy_sd_init(void);
void daisy_sd_deinit(void);
int daisy_sd_is_mounted(void);

DaisyResult daisy_spectral_load_sd(DaisySpectralCtx* ctx, const char* filename);
DaisyResult daisy_spectral_load_buffer(DaisySpectralCtx* ctx,
                                       const DaisySegmentQ15* data,
                                       uint32_t num_segments,
                                       uint32_t output_len);
void daisy_spectral_get_memory(const DaisySpectralCtx* ctx,
                               uint32_t* segments_bytes,
                               uint32_t* total_bytes,
                               uint32_t* available_bytes);

/* Parameters */
void daisy_spectral_set_stretch(DaisySpectralCtx* ctx, float stretch);
void daisy_spectral_set_amplitude(DaisySpectralCtx* ctx, float amplitude);
void daisy_spectral_set_params_adc(DaisySpectralCtx* ctx,
                                   uint16_t stretch_adc,
                                   uint16_t amplitude_adc);

/* Audio processing */
DAISY_HOT
uint32_t daisy_spectral_process_q15(DaisySpectralCtx* ctx,
                                    q15_t* out_left,
                                    q15_t* out_right,
                                    uint32_t num_samples);
DAISY_HOT
uint32_t daisy_spectral_process_interleaved(DaisySpectralCtx* ctx,
                                            q15_t* out_interleaved,
                                            uint32_t num_samples);

/* Playback control */
void daisy_spectral_reset(DaisySpectralCtx* ctx);
void daisy_spectral_seek(DaisySpectralCtx* ctx, uint32_t sample_pos);
int daisy_spectral_is_complete(const DaisySpectralCtx* ctx);
uint32_t daisy_spectral_get_position(const DaisySpectralCtx* ctx);
uint32_t daisy_spectral_get_duration(const DaisySpectralCtx* ctx);

/*
 * UART Protocol for remote parameter control
 * 
 * Message Format: [CMD] [DATA...] [CHECKSUM]
 *   CMD: Single byte command
 *   DATA: Command-specific payload (variable length)
 *   CHECKSUM: XOR of all preceding bytes
 * 
 * Commands:
 *   UART_CMD_PING       (0x01): Query device, responds with UART_RESP_PONG
 *   UART_CMD_SET_STRETCH(0x10): Set stretch (4-byte float, little-endian)
 *   UART_CMD_SET_AMP    (0x11): Set amplitude (4-byte float, little-endian)
 *   UART_CMD_RESET      (0x20): Reset playback position
 *   UART_CMD_SEEK       (0x21): Seek to position (4-byte uint32, little-endian)
 *   UART_CMD_GET_STATUS (0x30): Query playback status
 *   UART_CMD_LOAD_FILE  (0x40): Load file from SD (null-terminated string)
 * 
 * Responses:
 *   UART_RESP_OK        (0x00): Command succeeded
 *   UART_RESP_PONG      (0x01): Response to PING
 *   UART_RESP_ERR       (0xFF): Command failed (followed by error code)
 *   UART_RESP_STATUS    (0x30): Status response (position, duration, playing)
 */

#define DAISY_UART_CMD_PING         0x01
#define DAISY_UART_CMD_SET_STRETCH  0x10
#define DAISY_UART_CMD_SET_AMP      0x11
#define DAISY_UART_CMD_RESET        0x20
#define DAISY_UART_CMD_SEEK         0x21
#define DAISY_UART_CMD_GET_STATUS   0x30
#define DAISY_UART_CMD_LOAD_FILE    0x40

#define DAISY_UART_RESP_OK          0x00
#define DAISY_UART_RESP_PONG        0x01
#define DAISY_UART_RESP_ERR         0xFF
#define DAISY_UART_RESP_STATUS      0x30

#define DAISY_UART_MAX_MSG_LEN      64

typedef struct {
    uint32_t position;
    uint32_t duration;
    uint8_t  is_complete;
    uint8_t  is_loaded;
    uint16_t reserved;
} DaisyStatusResponse;

/* Process a single UART byte. Returns response length (0 if no response ready).
 * Call repeatedly with incoming UART bytes.
 * When response_len > 0, send response_buf back over UART. */
int daisy_uart_process_byte(DaisySpectralCtx* ctx, 
                            uint8_t byte,
                            uint8_t* response_buf,
                            uint8_t* response_len);

/* Reset UART protocol state (call on communication timeout) */
void daisy_uart_reset(void);

#ifdef __cplusplus
}

namespace daisy {

class SpectralEngineQ15 {
public:
    SpectralEngineQ15() : initialized_(false) {}
    ~SpectralEngineQ15() { Deinit(); }

    bool Init(uint32_t sample_rate = DAISY_SAMPLE_RATE) {
        if (daisy_spectral_init(&ctx_, sample_rate) == DAISY_OK) {
            initialized_ = true;
            return true;
        }
        return false;
    }

    void Deinit() {
        if (initialized_) {
            daisy_spectral_deinit(&ctx_);
            initialized_ = false;
        }
    }

    DaisyResult LoadFromSD(const char* filename) {
        return daisy_spectral_load_sd(&ctx_, filename);
    }

    void SetStretch(float stretch) { daisy_spectral_set_stretch(&ctx_, stretch); }
    void SetAmplitude(float amp) { daisy_spectral_set_amplitude(&ctx_, amp); }
    void SetParamsFromADC(uint16_t stretch, uint16_t amp) {
        daisy_spectral_set_params_adc(&ctx_, stretch, amp);
    }

    uint32_t Process(q15_t* left, q15_t* right, uint32_t num_samples) {
        return daisy_spectral_process_q15(&ctx_, left, right, num_samples);
    }
    uint32_t ProcessInterleaved(q15_t* out, uint32_t num_samples) {
        return daisy_spectral_process_interleaved(&ctx_, out, num_samples);
    }

    void Reset() { daisy_spectral_reset(&ctx_); }
    void Seek(uint32_t pos) { daisy_spectral_seek(&ctx_, pos); }
    bool IsComplete() const { return daisy_spectral_is_complete(&ctx_); }
    uint32_t GetPosition() const { return daisy_spectral_get_position(&ctx_); }
    uint32_t GetDuration() const { return daisy_spectral_get_duration(&ctx_); }

    void GetMemoryUsage(uint32_t* segs, uint32_t* total, uint32_t* avail) {
        daisy_spectral_get_memory(&ctx_, segs, total, avail);
    }
    
    /* Access underlying contexts */
    DaisySpectralCtx* GetContext() { return &ctx_; }
    SpectralArm32Ctx* GetSynthContext() { return &ctx_.synth; }

private:
    DaisySpectralCtx ctx_;
    bool initialized_;
};

} /* namespace daisy */

#endif /* __cplusplus */

#endif /* DAISY_SEED_SPECTRAL_H */
