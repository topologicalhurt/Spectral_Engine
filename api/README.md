# Spectral Engine API

Platform-specific API wrappers for spectral_engine.

## Structure

    daisy_seed/         Daisy Seed ARM embedded port

## Backends

    Desktop:   Float synthesis, OpenMP, vDSP, Metal/CUDA
    Embedded:  Q15 fixed-point, ARM Cortex-M7 optimized

## Daisy Seed

STM32H750 @ 480MHz, 64MB SDRAM, 96kHz audio.

## Usage

```c
#include "api/daisy_seed/daisy_seed_spectral.h"

DaisySpectralCtx ctx;
if (daisy_spectral_init(&ctx, DAISY_SAMPLE_RATE) != DAISY_OK) {
    return;
}

if (daisy_spectral_load_sd(&ctx, "segments.spq") != DAISY_OK) {
    daisy_spectral_deinit(&ctx);
    return;
}

q15_t left[DAISY_AUDIO_BLOCK_SIZE];
q15_t right[DAISY_AUDIO_BLOCK_SIZE];
daisy_spectral_process_q15(&ctx, left, right, DAISY_AUDIO_BLOCK_SIZE);
daisy_spectral_deinit(&ctx);
```

## Build

Run all commands from the repository root.

    make configure CMAKE_CONFIGURE_ARGS='-DSPECTRAL_DAISY_LIBDAISY_DIR=/path/to/libDaisy -DSPECTRAL_DAISY_DAISYSP_DIR=/path/to/DaisySP'
    make daisy

Optional Daisy example firmware:

    make configure CMAKE_CONFIGURE_ARGS='-DSPECTRAL_DAISY_LIBDAISY_DIR=/path/to/libDaisy -DSPECTRAL_DAISY_DAISYSP_DIR=/path/to/DaisySP -DSPECTRAL_DAISY_BUILD_EXAMPLE=ON'
    make daisy
    make daisy_example

## API status

Two versioned public surfaces (SemVer, currently 0.1.0 — pre-1.0, still subject
to change):

    spectral_engine/synth/api/spectral_synth.h   Desktop synthesis (float)
    api/daisy_seed/daisy_seed_spectral.h          Daisy Seed (Q15, Cortex-M7)


## Realtime notice

**Firstly, realtime operation is currently not verified or tested!**

This section will be updated in the future when it is.

It is impossible to know the frequency content of a file *a priori*. It could be very harmonically dense but it could also just as easily be sparse. There are ways of detecting and renormalizing files for assurance, but not without lossy quality tradeoffs which is not ideal. Because performance is so contingent on active segment count (and other choices of parameter like voice count or even the operating mode of the oscillator) **getting real-time assurance is hard.**

## Workflow

    1. Analyze on desktop:   ./build/bin/spectral_*_desktop input.wav 0 1.0 0 4096 128 -90 8 4
    2. Convert segments:     ./build/bin/convert_segments segments.bin daisy.spq
    3. Load on Daisy:        SD card or firmware embed
