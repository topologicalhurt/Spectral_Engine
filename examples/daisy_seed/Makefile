# examples/daisy_seed/Makefile - Example build for Daisy Seed
#
# This builds the example application. The API library is in api/daisy_seed/
#
# NOTE: spectral_synth_embedded.c is the ARM-optimized Q15 synthesis backend.
#       It can be compiled on desktop for testing but is cross-compiled here
#       for ARM Cortex-M7 (Daisy Seed).

# Project
TARGET = spectral_example

# Output directories (project root bin/)
BIN_DIR = ../../bin/daisy
BUILD_DIR = $(BIN_DIR)/example_obj

# Sources - just the example, API is built separately
CPP_SOURCES = example_spectral.cpp

# API library location
API_DIR = ../../api/daisy_seed
ENGINE_DIR = ../../spectral_engine

# libDaisy paths
LIBDAISY_DIR ?= $(DAISY_PATH)/libDaisy
DAISYSP_DIR ?= $(DAISY_PATH)/DaisySP

# Check for libDaisy
ifeq ($(wildcard $(LIBDAISY_DIR)),)
$(error libDaisy not found. Set DAISY_PATH or LIBDAISY_DIR environment variable)
endif

# System directories
SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# Additional includes
C_INCLUDES += -I$(API_DIR)
C_INCLUDES += -I$(ENGINE_DIR)

# Additional source files from API
C_SOURCES += $(API_DIR)/daisy_seed_spectral.c
C_SOURCES += $(ENGINE_DIR)/spectral_synth_embedded.c
C_SOURCES += $(ENGINE_DIR)/spectral_q15.c
C_SOURCES += $(ENGINE_DIR)/spectral_wavetable.c
C_SOURCES += $(ENGINE_DIR)/spectral_debug_embedded_arm.c

# Enable FatFS for SD card
C_DEFS += -DDAISY_HAS_FATFS=1

# 
#                    BUILD OPTIONS (set via command line)
# 

# Debug mode (enables ARM debug monitoring via JTAG/ITM)
ifeq ($(DEBUG),1)
C_DEFS += -DDAISY_DEBUG=1 -DSPECTRAL_DEBUG=1
OPT = -Og -g
else
OPT = -O2
endif

# Custom wavetable LUT support (build with WAVETABLE=1)
ifeq ($(WAVETABLE),1)
C_DEFS += -DSPECTRAL_USE_WAVETABLE_LUT=1
endif

# Custom timing overrides (example: make BLINK_PLAYING=500)
ifdef BLINK_PLAYING
C_DEFS += -DSPECTRAL_LED_BLINK_PLAYING_MS=$(BLINK_PLAYING)
endif

ifdef BLINK_DONE  
C_DEFS += -DSPECTRAL_LED_BLINK_DONE_MS=$(BLINK_DONE)
endif

ifdef ERROR_BLINK_ON
C_DEFS += -DSPECTRAL_ERROR_BLINK_ON_MS=$(ERROR_BLINK_ON)
endif

ifdef ERROR_BLINK_OFF
C_DEFS += -DSPECTRAL_ERROR_BLINK_OFF_MS=$(ERROR_BLINK_OFF)
endif

# 
#                              TARGETS
# 

.PHONY: all clean flash size

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

all: $(BIN_DIR) $(BUILD_DIR) $(BIN_DIR)/$(TARGET).elf $(BIN_DIR)/$(TARGET).hex $(BIN_DIR)/$(TARGET).bin

flash: $(BIN_DIR)/$(TARGET).bin
	dfu-util -a 0 -s 0x08000000:leave -D $<

size: $(BIN_DIR)/$(TARGET).elf
	$(SZ) $<
	@echo ""
	@echo "Memory regions:"
	@$(OBJDUMP) -h $< | grep -E '\.text|\.data|\.bss|\.sdram'

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)/$(TARGET).*

# Help
help:
	@echo "Spectral Engine - Daisy Seed Example"
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build firmware (default) -> ../../bin/daisy/"
	@echo "  flash  - Flash to Daisy Seed via DFU"
	@echo "  clean  - Remove build artifacts"
	@echo "  size   - Show memory usage"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1           - Build with debug symbols + JTAG monitoring"
	@echo "  WAVETABLE=1       - Enable wavetable LUT support"
	@echo "  BLINK_PLAYING=ms  - LED blink period while playing"
	@echo "  BLINK_DONE=ms     - LED blink period when done"
	@echo ""
	@echo "Debug Monitoring (DEBUG=1):"
	@echo "  CPU load, memory usage, real-time deadline tracking"
	@echo "  View via JTAG/SWO in Ozone or STM32CubeIDE"
	@echo ""
	@echo "Environment:"
	@echo "  DAISY_PATH        - Path to Daisy libraries"
