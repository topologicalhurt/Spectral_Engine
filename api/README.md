# Spectral Engine API

Platform-specific API wrappers for spectral_engine.

## Structure

    spectral_api.h      Platform-agnostic C API
    daisy_seed/         Daisy Seed ARM embedded port

## Backends

    Desktop:   Float synthesis, OpenMP, vDSP, Metal/CUDA
    Embedded:  Q15 fixed-point, ARM Cortex-M7 optimized

## Daisy Seed

STM32H750 @ 480MHz, 64MB SDRAM, 96kHz audio.

## Usage

```c
#include "api/spectral_api.h"

spectral_init(48000, SPECTRAL_BACKEND_AUTO);

SpectralStreamCtx ctx;
spectral_load_file(&ctx, "segments.bin");

SpectralParams params = { .stretch = 2.0f, .pitch = -5.0f };
spectral_set_params(&ctx, &params);
float output[256];
spectral_process_float(&ctx, output, 256);
spectral_unload(&ctx);
spectral_deinit();
```

## Build

    cd spectral_engine && make daisy
    Or
    cd api/daisy_seed && make


## Realtime notice

**Firstly, realtime operation is currently not verified or tested!**

This section will be updated in the future when it is.

It is impossible to know the frequency content of a file *a priori*. It could be very harmonically dense but it could also just as easily be sparse. There are ways of detecting and renormalizing files for assurance, but not without lossy quality tradeoffs which is not ideal. Because performance is so contingent on active segment count (and other choices of parameter like voice count or even the operating mode of the oscillator) **getting real-time assurance is hard.**

## Workflow

    1. Analyze on desktop:   ./bin/spectral input.wav 0 1.0 0 4096 128 -90 8 4
    2. Convert segments:     ./bin/convert_segments segments.bin daisy.bin
    3. Load on Daisy:        SD card or firmware embed
