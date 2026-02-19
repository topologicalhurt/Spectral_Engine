# Spectral Engine Examples

Platform-specific example applications.

## Structure

    daisy_seed/           Daisy Seed ARM example
      example_spectral.cpp

## Build

Run all commands from the repository root.

    make configure CMAKE_CONFIGURE_ARGS='-DSPECTRAL_DAISY_LIBDAISY_DIR=/path/to/libDaisy -DSPECTRAL_DAISY_DAISYSP_DIR=/path/to/DaisySP -DSPECTRAL_DAISY_BUILD_EXAMPLE=ON'
    make daisy_example
    make daisy_example_flash

## Daisy Seed Pins

    D15   Stretch pot (ADC)
    D16   Volume pot (ADC)
    D28   Play/Pause button
    D27   Reset button

SD card: place spectral.spq at root.
