# Daisy Seed Spectral Engine API Library Makefile
#
# This builds the Spectral Engine API for Daisy Seed hardware.
# For examples, see ../examples/daisy_seed/
#
# Requires the DaisySP library and ARM GCC toolchain.
#
# NOTE: spectral_synth_arm32.c is compiled here for ARM target.
#       The same code can be compiled on desktop for testing (see spectral_engine/Makefile)
#       but is optimized for ARM Cortex-M7 (SMLAD, cache prefetch, Q15 fixed-point).
#
# Usage:
#   make                  - Build the library objects -> ../../bin/daisy/
#   make clean            - Clean build artifacts
#   make debug            - Build with debug symbols + ARM debug monitoring
#   make WAVETABLE=1      - Build with wavetable LUT support

# 
#                        PROJECT CONFIGURATION
# 

# Target name (library)
TARGET = libspectral_daisy

# Output directories (project root bin/)
BIN_DIR = ../../bin/daisy
BUILD_DIR = $(BIN_DIR)/obj

# Sources (API only - no examples)
C_SOURCES = \
	daisy_seed_spectral.c \
	../../spectral_engine/spectral_synth_arm32.c \
	../../spectral_engine/spectral_q15.c \
	../../spectral_engine/spectral_wavetable.c \
	../../spectral_engine/spectral_debug_embedded_arm.c


CPP_SOURCES =

# Headers (for dependency tracking)
HEADERS = \
	daisy_seed_config.h \
	daisy_seed_spectral.h \
	../../spectral_engine/spectral_config.h \
	../../spectral_engine/spectral_q15.h \
	../../spectral_engine/oscillator.h \
	../../spectral_engine/spectral_wavetable.h \
	../../spectral_engine/spectral_synth_arm32.h \
	../../spectral_engine/spectral_debug_embedded_arm.h

# 
#                        DAISY LIBRARY PATHS
# 

# Path to libDaisy - adjust this to your installation
# Default assumes Daisy libraries are installed in standard location
LIBDAISY_DIR ?= $(HOME)/daisy/libDaisy
DAISYSP_DIR ?= $(HOME)/daisy/DaisySP

# Alternatively, use environment variables or pkg-config
ifdef DAISY_PATH
    LIBDAISY_DIR = $(DAISY_PATH)/libDaisy
    DAISYSP_DIR = $(DAISY_PATH)/DaisySP
endif

# 
#                           TOOLCHAIN
# 

# ARM GCC Toolchain
PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
CXX = $(PREFIX)g++
AS = $(PREFIX)gcc -x assembler-with-cpp
CP = $(PREFIX)objcopy
SZ = $(PREFIX)size
GDB = $(PREFIX)gdb
OBJDUMP = $(PREFIX)objdump
AR = $(PREFIX)ar

# 
#                        TARGET CONFIGURATION
# 

# STM32H750 (Daisy Seed MCU)
CPU = -mcpu=cortex-m7
FPU = -mfpu=fpv5-d16
FLOAT-ABI = -mfloat-abi=hard
MCU = $(CPU) -mthumb $(FPU) $(FLOAT-ABI)

# C/C++ defines
C_DEFS = \
	-DCORE_CM7 \
	-DSTM32H750xx \
	-DSTM32H750IB \
	-DARM_MATH_CM7 \
	-DUSE_HAL_DRIVER \
	-DSPECTRAL_PLATFORM_DAISY \
	-DDAISY_HAS_FATFS=1 \
	-DHSE_VALUE=16000000

# Debug build (enables ARM debug monitoring via JTAG/ITM)
ifeq ($(DEBUG), 1)
C_DEFS += -DDAISY_DEBUG=1 -DSPECTRAL_DEBUG=1
endif

# Wavetable LUT support (build with WAVETABLE=1)
ifeq ($(WAVETABLE), 1)
C_DEFS += -DSPECTRAL_USE_WAVETABLE_LUT=1
endif

# Custom timing overrides
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
#                        INCLUDE PATHS
# 

C_INCLUDES = \
	-I. \
	-I.. \
	-I$(LIBDAISY_DIR)/src \
	-I$(LIBDAISY_DIR)/src/per \
	-I$(LIBDAISY_DIR)/src/sys \
	-I$(LIBDAISY_DIR)/src/usbd \
	-I$(LIBDAISY_DIR)/src/usbh \
	-I$(LIBDAISY_DIR)/src/dev \
	-I$(LIBDAISY_DIR)/src/util \
	-I$(LIBDAISY_DIR)/Drivers/CMSIS/Include \
	-I$(LIBDAISY_DIR)/Drivers/CMSIS/Device/ST/STM32H7xx/Include \
	-I$(LIBDAISY_DIR)/Drivers/STM32H7xx_HAL_Driver/Inc \
	-I$(LIBDAISY_DIR)/Middlewares/ST/STM32_USB_Device_Library/Core/Inc \
	-I$(LIBDAISY_DIR)/Middlewares/ST/STM32_USB_Host_Library/Core/Inc \
	-I$(DAISYSP_DIR)/Source

# 
#                        COMPILER FLAGS
# 

# Optimization level
OPTIMIZE ?= 3

# Common flags
CFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES)
CFLAGS += -O$(OPTIMIZE) -Wall -Wextra -Wno-missing-field-initializers
CFLAGS += -fdata-sections -ffunction-sections
CFLAGS += -fno-exceptions -fno-rtti
CFLAGS += -fomit-frame-pointer -fno-unwind-tables -fno-asynchronous-unwind-tables

# DSP optimizations
CFLAGS += -ffast-math -funsafe-math-optimizations

# Size optimizations for embedded
CFLAGS += -fno-threadsafe-statics

# Generate dependency files
CFLAGS += -MMD -MP

# C++ specific flags
CXXFLAGS = $(CFLAGS) -std=c++17

# Debug flags
ifeq ($(DEBUG), 1)
CFLAGS += -g3 -gdwarf-2
endif

# Assembly flags
ASFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES) -x assembler-with-cpp

# 
#                          LINKER FLAGS
# 

# Linker script
LDSCRIPT = $(LIBDAISY_DIR)/core/STM32H750IB_flash.lds

# Libraries
LIBS = -ldaisy -lDaisySP -lc -lm -lnosys
LIBDIR = -L$(LIBDAISY_DIR)/build -L$(DAISYSP_DIR)/build

# Linker flags
LDFLAGS = $(MCU) -specs=nano.specs -specs=nosys.specs
LDFLAGS += -T$(LDSCRIPT) $(LIBDIR) $(LIBS)
LDFLAGS += -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,--print-memory-usage

# 
#                         BUILD DIRECTORY
# 

# Directories created in BIN_DIR/BUILD_DIR (already defined above)

# 
#                            TARGETS
# 

# Object files
C_OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
CPP_OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
OBJECTS = $(C_OBJECTS) $(CPP_OBJECTS)

# VPATH for finding sources in different directories
VPATH = .:../../spectral_engine

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Default target
all: $(BIN_DIR) $(BUILD_DIR) $(BIN_DIR)/$(TARGET).elf $(BIN_DIR)/$(TARGET).hex $(BIN_DIR)/$(TARGET).bin

# Binary outputs
$(BIN_DIR)/%.hex: $(BIN_DIR)/%.elf
	$(CP) -O ihex $< $@

$(BIN_DIR)/%.bin: $(BIN_DIR)/%.elf
	$(CP) -O binary -S $< $@

# ELF
$(BIN_DIR)/$(TARGET).elf: $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@
	$(SZ) $@

# C compilation
$(BUILD_DIR)/%.o: %.c $(HEADERS) | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# C++ compilation
$(BUILD_DIR)/%.o: %.cpp $(HEADERS) | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 
#                        FLASH TARGETS
# 

# Flash using DFU (USB bootloader)
flash: $(BIN_DIR)/$(TARGET).bin
	@echo "Entering bootloader mode..."
	@echo "Hold BOOT button, press RESET, release BOOT"
	dfu-util -a 0 -s 0x08000000:leave -D $<

# Flash using ST-Link (SWD)
flash-swd: $(BIN_DIR)/$(TARGET).elf
	openocd -f interface/stlink.cfg -f target/stm32h7x.cfg \
		-c "program $< verify reset exit"

# Debug with GDB + OpenOCD
debug-gdb: $(BIN_DIR)/$(TARGET).elf
	$(GDB) -ex "target remote :3333" -ex "monitor reset halt" $<

# 
#                        UTILITY TARGETS
# 

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Print size information
size: $(BIN_DIR)/$(TARGET).elf
	$(SZ) --format=berkeley $<
	@echo ""
	@echo "Section sizes:"
	$(SZ) --format=sysv $<

# Disassembly
disasm: $(BIN_DIR)/$(TARGET).elf
	$(OBJDUMP) -d -S $< > $(BIN_DIR)/$(TARGET).lst

# Print compiler info
info:
	@echo "============================================"
	@echo "Daisy Seed Spectral Engine Build"
	@echo "============================================"
	@echo "Target:     $(TARGET)"
	@echo "Output:     $(BIN_DIR)/"
	@echo "MCU:        STM32H750 (Cortex-M7 @ 480MHz)"
	@echo "Toolchain:  $(shell $(CC) --version | head -1)"
	@echo "libDaisy:   $(LIBDAISY_DIR)"
	@echo "DaisySP:    $(DAISYSP_DIR)"
	@echo "============================================"
	@echo ""
	@echo "Embedded Synthesis:"
	@echo "  spectral_synth_arm32.c - ARM M7 optimized Q15"
	@echo "  spectral_debug_embedded_arm.c - ARM debug monitoring (JTAG/ITM)"
	@echo "============================================"

# Check dependencies
check-deps:
	@echo "Checking for required tools..."
	@which $(CC) > /dev/null || (echo "ERROR: ARM GCC not found" && exit 1)
	@which dfu-util > /dev/null || echo "WARNING: dfu-util not found (needed for USB flashing)"
	@test -d $(LIBDAISY_DIR) || echo "WARNING: libDaisy not found at $(LIBDAISY_DIR)"
	@test -d $(DAISYSP_DIR) || echo "WARNING: DaisySP not found at $(DAISYSP_DIR)"
	@echo "Dependency check complete"

# 
#                     SEGMENT CONVERSION TOOL
# 

# Build desktop tool to convert segments to Daisy format
convert-tool:
	$(MAKE) -C ../../spectral_engine convert_segments

# Convert segments for Daisy
convert: convert-tool
	../../bin/convert_segments input.bin output.bin

# 
#                          HELP
# 

help:
	@echo "Daisy Seed Spectral Engine API Library"
	@echo ""
	@echo "  make           - Build library objects -> ../../bin/daisy/"
	@echo "  make clean     - Remove build files"
	@echo "  make size      - Show memory usage (requires linking)"
	@echo "  make info      - Show build configuration"
	@echo "  make check-deps - Verify required tools"
	@echo ""
	@echo "Build Options:"
	@echo "  DEBUG=1        - Enable debug symbols + ARM debug monitoring"
	@echo "  OPTIMIZE=0     - Disable optimization"
	@echo "  WAVETABLE=1    - Enable wavetable LUT support"
	@echo ""
	@echo "Debug Monitoring (DEBUG=1):"
	@echo "  - CPU cycle counting (DWT)"
	@echo "  - Memory usage tracking (SRAM/SDRAM)"
	@echo "  - Real-time deadline monitoring"
	@echo "  - ITM/SWO trace output via JTAG"
	@echo ""
	@echo "Timing Override Options (milliseconds):"
	@echo "  BLINK_PLAYING=ms  - LED blink period while playing (default: 250)"
	@echo "  BLINK_DONE=ms     - LED blink period when done (default: 100)"
	@echo "  ERROR_BLINK_ON=ms - Error blink on time (default: 100)"
	@echo "  ERROR_BLINK_OFF=ms - Error blink off time (default: 100)"
	@echo ""
	@echo "For examples, see: ../../examples/daisy_seed/"
	@echo ""
	@echo "Environment Variables:"
	@echo "  LIBDAISY_DIR   - Path to libDaisy"
	@echo "  DAISYSP_DIR    - Path to DaisySP"
	@echo "  DAISY_PATH     - Base path for Daisy libraries"

.PHONY: all clean size disasm info check-deps convert-tool convert help

# Include dependency files
-include $(OBJECTS:.o=.d)
