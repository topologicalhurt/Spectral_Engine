SPECTRAL

Real-time spectral analysis and resynthesis engine.

Platforms: Desktop (macOS/Linux), Metal GPU, CUDA GPU, ARM Cortex-M7

BUILD

Native build
```
  cd spectral_engine
  make deps && make
```

Cross-compiled build (currently supports arm cortex-M4/7 & daisy seed)
```
cd spectral_engine
make deps && make embedded <target>
```

Emulator build (for simulating cross-compiled code on desktop)
```
cd spectral_engine
make deps && make emulator <target>
```

Refer to make info for more

USAGE

  ./bin/spectral input.wav [timbre] [stretch] [pitch] [n_fft] [hop] [thresh] [threads] [backend]

  Timbres: 0=sine 1=saw 2=square 3=tri 4=asin 5=para 6=quant 7=pwm
  Backends: 0=auto 1=cpu 2=metal 3=cuda 4=export

FEATURES & WORKFLOW

The full-fledged desktop version builds with offline processing features & other QOL features not afforded on embedded targets, including (but not limited to):

STFT spectral analysis / resynthesis, wavelett transforms, essentially unlimited polyphony count, much larger sample sizes, complex oscillator design, exporting segment binaries, GPU acceleration, multi-threading support, full floats as opposed to q8.8 / q15 half-word format, 64bit support (& more) tools for sound design.

However, the embedded cross-compiled targets all build with a real-time assurance for the CMSIS platform (Arm Cortex A/M series), subject to certain parameter thresholds*, supporting dynamic (real-time!) resynthesis of imported segment binaries with up to 512 concurrent voices of polyphony. For a specific application refer to the daisy_seed api.

The currently supported workflow on supported embedded devices is to build the segment binary offline & then either bake it into the firmware or upload it onto the devices flash (DMA support coming soon!) I.e.
```
make && ../bin/spectral ../resources/motormouth_recites_shakespeare_he_saw_the_cat.wav 0 1 0 4096 128 -90 8 4 && make emulator daisy && ../bin/spectral_emulator_daisy segments.bin
```

It is currently being investigated whether or not it is possible to support real-time tracking on embedded devices* by running the firmware on top of an RTOS.

STRUCTURE

  spectral_engine/    Core analysis and synthesis
  api/                Platform-specific wrappers
  examples/           Example applications & Demo's
  tools/              Useful scripts or toolings
  resources/          Location for auxiliary files
  bin/                Compiled binaries

AUTHOR

  Connor Sinclair
  csin0659@uni.sydney.edu.au
