/* example_spectral.cpp - Daisy Seed Spectral Engine Example */
#include "daisy_seed.h"
#include "../../api/daisy_seed/daisy_seed_spectral.h"

using namespace daisy;

static DaisySeed hw;
static DaisySpectralCtx spectral_ctx;

static volatile bool g_is_playing = false;
static volatile bool g_play_pressed = false;
static volatile bool g_reset_pressed = false;

static Switch switch_play;
static Switch switch_reset;

enum AdcChannel { ADC_STRETCH = 0, ADC_VOLUME, ADC_NUM_CHANNELS };

static uint32_t g_led_counter = 0;
static bool g_led_state = false;

void TimerCallback(void* data) {
    (void)data;
    
    switch_play.Debounce();
    switch_reset.Debounce();
    
    if (switch_play.RisingEdge()) g_play_pressed = true;
    if (switch_reset.RisingEdge()) g_reset_pressed = true;
    
    uint32_t blink_period;
    if (g_is_playing && !daisy_spectral_is_complete(&spectral_ctx)) {
        blink_period = SPECTRAL_LED_BLINK_PLAYING_MS;
    } else if (daisy_spectral_is_complete(&spectral_ctx)) {
        blink_period = SPECTRAL_LED_BLINK_DONE_MS;
    } else {
        blink_period = 0;
        g_led_state = false;
    }
    
    if (blink_period > 0) {
        if (++g_led_counter >= blink_period) {
            g_led_state = !g_led_state;
            g_led_counter = 0;
        }
    }
    
    hw.SetLed(g_led_state);
}

void AudioCallback(AudioHandle::InterleavingInputBuffer in,
                   AudioHandle::InterleavingOutputBuffer out,
                   size_t size) {
    (void)in;
    
    uint16_t stretch_adc = hw.adc.Get(ADC_STRETCH) >> 4;
    uint16_t volume_adc = hw.adc.Get(ADC_VOLUME) >> 4;
    daisy_spectral_set_params_adc(&spectral_ctx, stretch_adc, volume_adc);
    
    if (!g_is_playing || daisy_spectral_is_complete(&spectral_ctx)) {
        for (size_t i = 0; i < size * 2; i++) out[i] = 0.0f;
        return;
    }
    
    q15_t q15_buffer[DAISY_AUDIO_BLOCK_SIZE * 2];
    daisy_spectral_process_interleaved(&spectral_ctx, q15_buffer, size);
    
    for (size_t i = 0; i < size * 2; i++) out[i] = Q15_TO_FLOAT(q15_buffer[i]);
}

void ErrorBlink(uint32_t count, 
                uint32_t on_ms = SPECTRAL_ERROR_BLINK_ON_MS,
                uint32_t off_ms = SPECTRAL_ERROR_BLINK_OFF_MS,
                uint32_t pause_ms = SPECTRAL_ERROR_BLINK_PAUSE_MS) {
    while (1) {
        for (uint32_t i = 0; i < count; i++) {
            hw.SetLed(true);
            System::Delay(on_ms);
            hw.SetLed(false);
            System::Delay(off_ms);
        }
        System::Delay(pause_ms);
    }
}

int main(void) {
    hw.Configure();
    hw.Init();
    hw.SetAudioBlockSize(DAISY_AUDIO_BLOCK_SIZE);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    
    AdcChannelConfig adc_config[ADC_NUM_CHANNELS];
    adc_config[ADC_STRETCH].InitSingle(hw.GetPin(DAISY_PIN_STRETCH_POT));
    adc_config[ADC_VOLUME].InitSingle(hw.GetPin(DAISY_PIN_VOLUME_POT));
    hw.adc.Init(adc_config, ADC_NUM_CHANNELS);
    hw.adc.Start();
    
    switch_play.Init(hw.GetPin(DAISY_PIN_PLAY_BTN), 1000.0f,
                     Switch::TYPE_MOMENTARY, Switch::POLARITY_INVERTED);
    switch_reset.Init(hw.GetPin(DAISY_PIN_RESET_BTN), 1000.0f,
                      Switch::TYPE_MOMENTARY, Switch::POLARITY_INVERTED);
    
    DaisyResult result = daisy_spectral_init(&spectral_ctx, DAISY_SAMPLE_RATE);
    if (result != DAISY_OK) ErrorBlink(2);
    
    result = daisy_sd_init();
    #if DAISY_DEBUG
    if (result != DAISY_OK) hw.PrintLine("SD init failed: %d", result);
    #endif
    
    result = daisy_spectral_load_sd(&spectral_ctx, "0:/spectral.spq");
    #if DAISY_DEBUG
    if (result != DAISY_OK) {
        hw.PrintLine("Failed to load spectral.spq: %d", result);
    } else {
        hw.PrintLine("Loaded %d segments (%.2fs)", spectral_ctx.synth.num_segments,
                     spectral_ctx.synth.output_length / (float)DAISY_SAMPLE_RATE);
    }
    #endif
    
    TimerHandle timer;
    TimerHandle::Config timer_cfg;
    timer_cfg.periph = TimerHandle::Config::Peripheral::TIM_5;
    timer_cfg.dir = TimerHandle::Config::CounterDir::UP;
    timer.Init(timer_cfg);
    timer.SetCallback(TimerCallback);
    timer.SetPeriod(1000);
    timer.Start();
    
    hw.StartAudio(AudioCallback);
    
    while (1) {
        if (g_play_pressed) {
            g_play_pressed = false;
            g_is_playing = !g_is_playing;
            if (g_is_playing && daisy_spectral_is_complete(&spectral_ctx)) {
                daisy_spectral_reset(&spectral_ctx);
            }
        }
        
        if (g_reset_pressed) {
            g_reset_pressed = false;
            daisy_spectral_reset(&spectral_ctx);
        }
        
        __WFI();
    }
}
