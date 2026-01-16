# Spectral Engine Examples

Platform-specific example applications.

## Structure

    daisy_seed/           Daisy Seed ARM example
      example_spectral.cpp
      Makefile

## Build

    cd daisy_seed
    make
    make flash

## Daisy Seed Pins

    D15   Stretch pot (ADC)
    D16   Volume pot (ADC)
    D28   Play/Pause button
    D27   Reset button

SD card: place spectral.spq at root.
