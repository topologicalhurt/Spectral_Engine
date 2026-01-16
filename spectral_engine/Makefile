# 
#                         SPECTRAL ENGINE MAKEFILE
# 
#
# BUILD TARGETS:
# --------------
#   make / make desktop     Full desktop build (Metal/CUDA, FFT analysis)
#   make emulator           Desktop emulation of Q15 embedded synth path
#   make emulator daisy     Desktop emulation with Daisy Seed config
#   make cuda               Standalone CUDA-only build
#   make embedded arm       Cross-compile: ARM Q15 with FFT analysis
#   make embedded arm_restricted  Cross-compile: ARM Q15, NO analysis
#   make daisy              Cross-compile for Daisy Seed (via api/daisy_seed)
#   make clean              Remove all build artifacts
#   make info               Show build configuration matrix
#
# BUILD MATRIX:
# -------------
# Target                    | Analysis | Synth Backend      | GPU  | Sample Type | Runs On
# --------------------------|----------|--------------------| -----|-------------|--------
# desktop (default)         | YES      | spectral_synth_cpu | YES  | float       | Desktop
# emulator                  | YES      | spectral_synth_emulator | NO | Q15->float | Desktop
# emulator daisy            | NO       | spectral_synth_emulator | NO | Q15->float | Desktop
# cuda                      | YES      | CUDA standalone    | YES  | float       | Desktop
# embedded arm              | YES      | spectral_synth_embedded | NO | Q15        | ARM
# embedded arm_restricted   | NO       | spectral_synth_embedded | NO | Q15        | ARM
# daisy                     | NO       | (via api/)         | NO   | Q15         | ARM
#
# COMPILE FLAGS REFERENCE:
# ------------------------
# Flag                      | Description
# --------------------------|--------------------------------------------------
# SPECTRAL_EMBEDDED         | 1=Q15 fixed-point, 0=float (default)
# SPECTRAL_RESTRICTED_MODE  | 1=No FFT analysis/GPU, 0=full features (default)
# SPECTRAL_EMBEDDED_EMULATION | 1=Desktop emulating Q15 path (testing)
# SPECTRAL_USE_EMBEDDED_SYNTH | 1=Use embedded synth instead of CPU synth
# SPECTRAL_COMPACT_SEG      | 1=32-byte segments, 0=64-byte (default)
# SPECTRAL_NO_PERF          | 1=Disable perf tracking, 0=enabled (default)
# SPECTRAL_MAX_SEGS         | Maximum segment count (0=unlimited)
# SPECTRAL_DEBUG_RESTRICTED | 1=Enable debug profiling in restricted mode
# USE_CUDA                  | 1=Enable CUDA backend (auto-detected)
#
# OUTPUT DIRECTORY:
#   All binaries placed in ../bin/ (project root)
#
# 

# 
#                              CONFIGURATION
# 

# Output directories
BIN_DIR     := ../bin
OBJ_DIR     := $(BIN_DIR)/obj
OBJ_ARM_DIR := $(BIN_DIR)/obj_arm
OBJ_RES_DIR := $(BIN_DIR)/obj_restricted

# Binary names
TARGET_DESKTOP    := spectral
TARGET_ARM        := spectral_arm
TARGET_RESTRICTED := spectral_arm_restricted
TARGET_CUDA       := spectral_cuda

# Platform detection
UNAME_S := $(shell uname -s)
NVCC    := $(shell command -v nvcc 2>/dev/null)

# 
#                      COMPILER VERSION CHECK
# 
# Minimum required versions:
#   - clang 14+ (macOS)
#   - gcc 11+ (Linux)
#
# To override version check: make SKIP_VERSION_CHECK=1

ifndef SKIP_VERSION_CHECK
ifeq ($(UNAME_S),Darwin)
    # Check clang version (minimum 14)
    CLANG_VERSION := $(shell clang --version 2>/dev/null | head -1 | sed 's/.*version \([0-9]*\).*/\1/')
    CLANG_VERSION_OK := $(shell [ "$(CLANG_VERSION)" -ge 14 ] 2>/dev/null && echo 1 || echo 0)
    ifeq ($(CLANG_VERSION_OK),0)
        $(error Clang version $(CLANG_VERSION) is too old. Minimum required: 14. \
                Update with: xcode-select --install or brew install llvm. \
                To skip this check: make SKIP_VERSION_CHECK=1)
    endif
else
    # Check gcc version (minimum 11)
    GCC_VERSION := $(shell gcc -dumpversion 2>/dev/null | cut -d. -f1)
    GCC_VERSION_OK := $(shell [ "$(GCC_VERSION)" -ge 11 ] 2>/dev/null && echo 1 || echo 0)
    ifeq ($(GCC_VERSION_OK),0)
        $(error GCC version $(GCC_VERSION) is too old. Minimum required: 11. \
                Update with: sudo apt install gcc-11 or equivalent. \
                To skip this check: make SKIP_VERSION_CHECK=1)
    endif
endif
endif

# 
#                            SOURCE FILE GROUPS
# 

# Core sources (always included)
SRCS_CORE := oscillator.c spectral_common.c spectral_config.c \
			 spectral_error.c spectral_in.c spectral_out.c \
			 spectral_segment_parser.c spectral_segment_pool.c \
			 spectral_segment_mt.c spectral_synth_internal.c spectral_wavetable.c

# Performance/debug
SRCS_PERF := spectral_perf.c
SRCS_DEBUG_EMBEDDED := spectral_debug_embedded_arm.c

# Analysis module (requires FFT - desktop only)
SRCS_ANALYSIS := spectral_analysis.c

# Synthesis backends (mutually exclusive in most builds)
SRCS_SYNTH_CPU      := spectral_synth_cpu.c       # Desktop floating-point
SRCS_SYNTH_EMBEDDED := spectral_synth_embedded.c spectral_q15.c  # ARM Q15 optimized
SRCS_SYNTH_EMULATOR := spectral_synth_emulator.c  # Desktop wrapper for Q15 synth

# GPU backends (desktop only)
SRCS_METAL := spectral_synth_metal.m
SRCS_CUDA  := spectral_synth_cuda.cu

# All headers (for dependency tracking)
HEADERS := spectral_common.h spectral_config.h spectral_error.h spectral_analysis.h \
           spectral_synth.h spectral_synth_internal.h spectral_synth_embedded.h \
           spectral_wavetable.h spectral_perf.h spectral_q15.h spectral_debug_embedded_arm.h \
           spectral_segment_pool.h spectral_segment_mt.h \
           oscillator.h spectral_utils.h spectral_segment_parser.h spectral_io.h \
           spectral_cli.h spectral_cli_pipeline.h

# 
#                         PLATFORM-SPECIFIC CONFIG
# 
# Compiler flags optimized for real-time audio synthesis:
#   - Aggressive floating-point optimization (assumes no NaN/Inf)
#   - SIMD vectorization for batch sample processing
#   - Loop unrolling for inner synthesis loops
#   - LTO for cross-function inlining
#   - Native CPU tuning for maximum throughput
#

# Common audio DSP optimization flags (both platforms)
AUDIO_OPT_FLAGS := -ffast-math -fno-signed-zeros -fno-trapping-math \
                   -fassociative-math -freciprocal-math -ffp-contract=fast \
                   -funroll-loops -ftree-vectorize -fomit-frame-pointer

ifeq ($(UNAME_S),Darwin)
    # macOS configuration (clang 14+)
    CC := clang
    
    # Homebrew prefix detection (Apple Silicon vs Intel)
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
    OMP_PREFIX     := $(BREW_PREFIX)/opt/libomp
    SNDFILE_PREFIX := $(BREW_PREFIX)/opt/libsndfile
    
    # Verify dependencies exist
    ifeq ($(wildcard $(OMP_PREFIX)/lib/libomp.dylib),)
        $(warning libomp not found at $(OMP_PREFIX). Install with: brew install libomp)
    endif
    ifeq ($(wildcard $(SNDFILE_PREFIX)/lib/libsndfile.dylib),)
        $(warning libsndfile not found at $(SNDFILE_PREFIX). Install with: brew install libsndfile)
    endif
    
    # Base flags with audio DSP optimizations
    # -flto=thin: Thin LTO for faster builds with cross-module optimization
    # -fvectorize: Enable loop vectorization (clang default with -O3, explicit for clarity)
    # -fslp-vectorize: Enable superword-level parallelism (pack scalar ops into SIMD)
    CFLAGS_BASE := -O3 -march=native -mtune=native $(AUDIO_OPT_FLAGS) \
                   -flto=thin -fvectorize -fslp-vectorize \
                   -Wall -Wextra -Wno-unknown-pragmas \
                   -I$(OMP_PREFIX)/include -I$(SNDFILE_PREFIX)/include \
                   -Xpreprocessor -fopenmp
    
    OBJCFLAGS := $(CFLAGS_BASE) -fobjc-arc
    
    # LTO requires passing flags to linker
    LDFLAGS_BASE := -flto=thin -lm -L$(OMP_PREFIX)/lib -lomp -L$(SNDFILE_PREFIX)/lib -lsndfile
    LDFLAGS_GPU  := $(LDFLAGS_BASE) -framework Accelerate -framework Metal -framework Foundation
    LDFLAGS_CPU  := $(LDFLAGS_BASE) -framework Accelerate
    
    HAS_METAL_PLATFORM := 1
else
    # Linux configuration (gcc 11+)
    CC := gcc
    
    # Base flags with audio DSP optimizations
    # -flto=auto: Full LTO with automatic parallelism
    # -floop-nest-optimize: Optimize nested loops (common in synthesis)
    # -fgraphite-identity: Enable Graphite loop optimizer
    # -fipa-pta: Interprocedural pointer analysis for better aliasing
    CFLAGS_BASE  := -std=c11 -O3 -march=native -mtune=native $(AUDIO_OPT_FLAGS) \
                    -flto=auto -floop-nest-optimize -fgraphite-identity -fipa-pta \
                    -Wall -Wextra -Wno-unknown-pragmas -fopenmp
    
    # LTO requires passing flags to linker
    LDFLAGS_BASE := -flto=auto -lm -lsndfile -lfftw3f -fopenmp
    LDFLAGS_GPU  := $(LDFLAGS_BASE)
    LDFLAGS_CPU  := $(LDFLAGS_BASE)
    
    HAS_METAL_PLATFORM := 0
    
    # CUDA support on Linux
    ifneq ($(NVCC),)
        CFLAGS_BASE  += -DUSE_CUDA
        # CUDA optimization: use fast math, enable FMA, tune for architecture
        NVCC_FLAGS   := -O3 -use_fast_math --fmad=true \
                        -Xcompiler -fPIC -Xcompiler -ffast-math -Xcompiler -funroll-loops
        CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -lcudart
        LDFLAGS_GPU  += $(CUDA_LDFLAGS)
    endif
endif

# 
#                           TARGET: desktop (default)
# 
# Full desktop build with all features:
#   - FFT analysis (spectral_analysis.c)
#   - Floating-point synthesis (spectral_synth_cpu.c)
#   - Metal GPU backend (macOS)
#   - CUDA GPU backend (Linux, if available)
#   - Performance tracking

# CLI sources (desktop/emulator only, not cross-compiled)
SRCS_CLI := spectral_cli.c spectral_cli_pipeline.c spectral_utils.c

DESKTOP_CFLAGS := $(CFLAGS_BASE)
DESKTOP_SRCS_C := main.c $(SRCS_CLI) $(SRCS_ANALYSIS) $(SRCS_SYNTH_CPU) $(SRCS_CORE) $(SRCS_PERF)

ifeq ($(UNAME_S),Darwin)
    DESKTOP_SRCS_OBJC := $(SRCS_METAL)
    DESKTOP_OBJS := $(addprefix $(OBJ_DIR)/,$(DESKTOP_SRCS_C:.c=.o)) \
                    $(addprefix $(OBJ_DIR)/,$(DESKTOP_SRCS_OBJC:.m=.o))
    DESKTOP_LDFLAGS := $(LDFLAGS_GPU)
else
    DESKTOP_OBJS := $(addprefix $(OBJ_DIR)/,$(DESKTOP_SRCS_C:.c=.o))
    ifneq ($(NVCC),)
        DESKTOP_OBJS += $(OBJ_DIR)/spectral_synth_cuda.o
    endif
    DESKTOP_LDFLAGS := $(LDFLAGS_GPU)
endif

# Default target
all: desktop

desktop: $(BIN_DIR)/$(TARGET_DESKTOP)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_DESKTOP)"
	@echo "================================================"
	@echo "Full desktop build:"
	@echo "  - FFT analysis enabled"
	@echo "  - Floating-point synthesis"
ifeq ($(UNAME_S),Darwin)
	@echo "  - Metal GPU backend"
endif
ifneq ($(NVCC),)
	@echo "  - CUDA GPU backend"
endif
	@echo ""
	@echo "Usage: $(BIN_DIR)/$(TARGET_DESKTOP) input.wav [options]"
	@echo "================================================"

$(BIN_DIR)/$(TARGET_DESKTOP): $(DESKTOP_OBJS) | $(BIN_DIR)
	$(CC) $(DESKTOP_OBJS) -o $@ $(DESKTOP_LDFLAGS)

# 
#                            TARGET: emulator
# 
# Desktop emulation of Q15 embedded synthesis path:
#   - FFT analysis enabled (can analyze WAV files)
#   - Q15 fixed-point synthesis via emulator wrapper
#   - NO GPU backends
#   - Runs on desktop, uses same math as ARM target
#
# Variants:
#   make emulator           Generic embedded emulator (Cortex-M7 defaults)
#   make emulator daisy     Daisy Seed specific emulator
#
# Use this to test/debug the embedded Q15 synthesis path on desktop
# with full tooling support. Output is identical to ARM hardware.

TARGET_EMULATOR := spectral_emulator
OBJ_EMU_DIR     := $(BIN_DIR)/obj_emulator

EMULATOR_CFLAGS := $(CFLAGS_BASE) -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_USE_EMBEDDED_SYNTH=1 \
                   -DSPECTRAL_EMBEDDED_EMULATION=1
EMULATOR_SRCS_C := main.c $(SRCS_CLI) $(SRCS_ANALYSIS) $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) \
                   $(SRCS_CORE) $(SRCS_PERF)
EMULATOR_OBJS   := $(addprefix $(OBJ_EMU_DIR)/,$(EMULATOR_SRCS_C:.c=.o))

# Emulator target dispatcher
.PHONY: emulator
emulator:
	@if [ "$(word 2,$(MAKECMDGOALS))" = "daisy" ]; then \
		$(MAKE) --no-print-directory emulator-daisy; \
	else \
		$(MAKE) --no-print-directory emulator-generic; \
	fi

# Generic emulator (default Cortex-M7 specs)
.PHONY: emulator-generic
emulator-generic: $(BIN_DIR)/$(TARGET_EMULATOR)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_EMULATOR)"
	@echo "================================================"
	@echo "Q15 Emulator Build (runs on desktop):"
	@echo "  - FFT analysis enabled"
	@echo "  - Q15 fixed-point synthesis (emulated)"
	@echo "  - Same math as ARM target"
	@echo "  - Default target: Cortex-M7 @ 480MHz"
	@echo ""
	@echo "Usage: $(BIN_DIR)/$(TARGET_EMULATOR) input.wav [options]"
	@echo "================================================"

# Daisy Seed emulator target
TARGET_EMULATOR_DAISY := spectral_emulator_daisy
OBJ_EMU_DAISY_DIR     := $(BIN_DIR)/obj_emulator_daisy

EMULATOR_DAISY_CFLAGS := $(CFLAGS_BASE) -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_USE_EMBEDDED_SYNTH=1 \
                         -DSPECTRAL_EMBEDDED_EMULATION=1 -DSPECTRAL_RESTRICTED_MODE=1 \
                         -DEMULATOR_TARGET_DAISY=1
EMULATOR_DAISY_SRCS_C := main.c $(SRCS_CLI) $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) $(SRCS_CORE) $(SRCS_PERF)
EMULATOR_DAISY_OBJS   := $(addprefix $(OBJ_EMU_DAISY_DIR)/,$(EMULATOR_DAISY_SRCS_C:.c=.o))

.PHONY: emulator-daisy
emulator-daisy: $(BIN_DIR)/$(TARGET_EMULATOR_DAISY)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_EMULATOR_DAISY)"
	@echo "================================================"
	@echo "Daisy Seed Emulator Build (runs on desktop):"
	@echo "  - NO FFT analysis (RESTRICTED_MODE)"
	@echo "  - Q15 fixed-point synthesis (emulated)"
	@echo "  - Single-threaded (matches Daisy)"
	@echo "  - Target: STM32H750 Cortex-M7 @ 480MHz"
	@echo "  - Simulates: 48kHz, 512KB SRAM"
	@echo ""
	@echo "Usage: $(BIN_DIR)/$(TARGET_EMULATOR_DAISY) segments.bin [options]"
	@echo "================================================"

$(BIN_DIR)/$(TARGET_EMULATOR): $(EMULATOR_OBJS) | $(BIN_DIR)
	$(CC) $(EMULATOR_OBJS) -o $@ $(LDFLAGS_CPU)

$(BIN_DIR)/$(TARGET_EMULATOR_DAISY): $(EMULATOR_DAISY_OBJS) | $(BIN_DIR)
	$(CC) $(EMULATOR_DAISY_OBJS) -o $@ $(LDFLAGS_CPU)

# Emulator object files (generic)
$(OBJ_EMU_DIR)/%.o: %.c $(HEADERS) | $(OBJ_EMU_DIR)
	$(CC) $(EMULATOR_CFLAGS) -c $< -o $@

# Emulator object files (Daisy)
$(OBJ_EMU_DAISY_DIR)/%.o: %.c $(HEADERS) | $(OBJ_EMU_DAISY_DIR)
	$(CC) $(EMULATOR_DAISY_CFLAGS) -c $< -o $@

# 
#                          TARGET: embedded arm
# 
# Cross-compile ARM Q15 synthesis with FFT analysis:
#   - FFT analysis enabled
#   - Q15 fixed-point synthesis (spectral_synth_embedded.c)
#   - NO GPU backends
#   - Binary targets ARM architecture (won't run on desktop)
#
# NOTE: This is for CROSS-COMPILATION. Use 'make emulator' to test
# the Q15 path on desktop.

# ARM flags - embedded Q15 with analysis (for testing)
ARM_EMBEDDED_FLAGS := -ffast-math -funsafe-math-optimizations \
                      -fdata-sections -ffunction-sections \
                      -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_USE_EMBEDDED_SYNTH=1

ARM_CFLAGS := $(CFLAGS_BASE) $(ARM_EMBEDDED_FLAGS)
ARM_SRCS_C := main.c $(SRCS_ANALYSIS) $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) $(SRCS_CORE) $(SRCS_PERF)
ARM_OBJS   := $(addprefix $(OBJ_ARM_DIR)/,$(ARM_SRCS_C:.c=.o))

# 
#                       TARGET: embedded arm_restricted
# 
# Cross-compile restricted ARM mode (Daisy Seed compatible):
#   - NO FFT analysis (loads pre-analyzed segments)
#   - Q15 fixed-point synthesis
#   - NO GPU backends
#   - Binary targets ARM architecture (won't run on desktop)
#
# NOTE: This is for CROSS-COMPILATION. Use 'make emulator' to test
# the Q15 path on desktop.

# Aggressive optimization flags for embedded ARM targets
# These mirror the flags used in api/daisy_seed/Makefile for Cortex-M7
EMBEDDED_OPT_FLAGS := -ffast-math -funsafe-math-optimizations \
                      -fdata-sections -ffunction-sections \
                      -fno-exceptions -fno-threadsafe-statics \
                      -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_RESTRICTED_MODE=1 \
                      -DSPECTRAL_USE_EMBEDDED_SYNTH=1 -DSPECTRAL_NO_PERF=1

RESTRICTED_CFLAGS := $(CFLAGS_BASE) $(EMBEDDED_OPT_FLAGS)
RESTRICTED_SRCS_C := main.c $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) $(SRCS_CORE)
RESTRICTED_OBJS   := $(addprefix $(OBJ_RES_DIR)/,$(RESTRICTED_SRCS_C:.c=.o))

# Embedded target dispatcher
.PHONY: embedded
embedded:
	@if [ -z "$(filter arm arm_restricted,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make embedded <target>"; \
		echo ""; \
		echo "Available targets (CROSS-COMPILATION for ARM):"; \
		echo "  arm            - Q15 synth + FFT analysis"; \
		echo "  arm_restricted - Q15 synth, NO analysis (Daisy Seed)"; \
		echo ""; \
		echo "To test Q15 path on desktop, use: make emulator"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make embedded arm"; \
		echo "  make embedded arm_restricted"; \
		exit 1; \
	fi

# ARM target (with analysis)
.PHONY: arm
arm: $(BIN_DIR)/$(TARGET_ARM)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_ARM)"
	@echo "================================================"
	@echo "ARM Cross-Compile (with analysis):"
	@echo "  - FFT analysis enabled"
	@echo "  - Q15 fixed-point synthesis"
	@echo "  - Target: ARM architecture"
	@echo ""
	@echo "NOTE: Binary won't run on this machine."
	@echo "      Use 'make emulator' to test Q15 path on desktop."
	@echo "================================================"

$(BIN_DIR)/$(TARGET_ARM): $(ARM_OBJS) | $(BIN_DIR)
	$(CC) $(ARM_OBJS) -o $@ $(LDFLAGS_CPU)

# ARM restricted target (no analysis, Daisy-like)
.PHONY: arm_restricted
arm_restricted: $(BIN_DIR)/$(TARGET_RESTRICTED)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_RESTRICTED)"
	@echo "================================================"
	@echo "ARM Cross-Compile Restricted (Daisy Seed):"
	@echo "  - NO FFT analysis (load pre-analyzed segments)"
	@echo "  - Q15 fixed-point synthesis"
	@echo "  - Target: ARM architecture"
	@echo ""
	@echo "NOTE: Binary won't run on this machine."
	@echo "      Use 'make emulator' to test Q15 path on desktop."
	@echo "================================================"

$(BIN_DIR)/$(TARGET_RESTRICTED): $(RESTRICTED_OBJS) | $(BIN_DIR)
	$(CC) $(RESTRICTED_OBJS) -o $@ $(LDFLAGS_CPU)

# 
#                             TARGET: cuda
# 
# Standalone CUDA build (Linux only)

cuda: $(BIN_DIR)
ifeq ($(NVCC),)
	@echo "Error: nvcc not found. Please install CUDA toolkit:"
	@echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
	@echo "  Or download from: https://developer.nvidia.com/cuda-downloads"
	@exit 1
else
	$(NVCC) -O3 -arch=sm_70 spectral_cuda.cu -o $(BIN_DIR)/$(TARGET_CUDA) -lsndfile
	@echo "Built $(BIN_DIR)/$(TARGET_CUDA)"
endif

# 
#                             TARGET: daisy
# 
# Cross-compile for Daisy Seed hardware (implies arm_restricted mode)
# Delegates to api/daisy_seed which uses ARM GCC toolchain
#
# Requires: arm-none-eabi-gcc toolchain
#
# Note: "daisy" is also used as a modifier for "make emulator daisy"
# When used as modifier, daisy-build is skipped (emulator-daisy handles it)

ARM_GCC := $(shell command -v arm-none-eabi-gcc 2>/dev/null)

# Check if daisy is being used as emulator modifier
DAISY_IS_MODIFIER := $(if $(filter emulator,$(word 1,$(MAKECMDGOALS))),yes,)

.PHONY: daisy daisy-flash daisy-clean
daisy:
ifndef DAISY_IS_MODIFIER
	@if [ -z "$(ARM_GCC)" ]; then \
		echo "============================================="; \
		echo "ERROR: ARM toolchain not found"; \
		echo "============================================="; \
		echo ""; \
		echo "Install arm-none-eabi-gcc:"; \
		echo "  macOS:   brew install arm-none-eabi-gcc"; \
		echo "  Ubuntu:  sudo apt install gcc-arm-none-eabi"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Building Daisy Seed firmware (arm_restricted mode)..."
	$(MAKE) -C ../api/daisy_seed all
endif

daisy-flash:
	$(MAKE) -C ../api/daisy_seed flash

daisy-clean:
	$(MAKE) -C ../api/daisy_seed clean

# 
#                           OBJECT FILE RULES
# 

# Create directories
$(BIN_DIR) $(OBJ_DIR) $(OBJ_ARM_DIR) $(OBJ_RES_DIR) $(OBJ_EMU_DIR) $(OBJ_EMU_DAISY_DIR):
	@mkdir -p $@

# Desktop object files
$(OBJ_DIR)/%.o: %.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(DESKTOP_CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.m $(HEADERS) | $(OBJ_DIR)
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(OBJ_DIR)/spectral_synth_cuda.o: spectral_synth_cuda.cu $(HEADERS) | $(OBJ_DIR)
ifneq ($(NVCC),)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

# ARM object files (with analysis)
$(OBJ_ARM_DIR)/%.o: %.c $(HEADERS) | $(OBJ_ARM_DIR)
	$(CC) $(ARM_CFLAGS) -c $< -o $@

# Restricted object files (no analysis)
$(OBJ_RES_DIR)/%.o: %.c $(HEADERS) | $(OBJ_RES_DIR)
	$(CC) $(RESTRICTED_CFLAGS) -c $< -o $@

# 
#                              UTILITIES
# 

.PHONY: debug clean info deps help

debug: DESKTOP_CFLAGS += -g -O0 -DDEBUG
debug: ARM_CFLAGS += -g -O0 -DDEBUG
debug: RESTRICTED_CFLAGS += -g -O0 -DDEBUG
debug: clean all

clean:
	rm -rf $(BIN_DIR) out_c.wav segments.bin
	@echo "Cleaned all build artifacts."

help:
	@echo "Spectral Engine Build System"
	@echo "============================"
	@echo ""
	@echo "Primary Targets:"
	@echo "  make                    Build full desktop version"
	@echo "  make desktop            Same as above"
	@echo "  make emulator           Q15 emulator (generic Cortex-M7)"
	@echo "  make emulator daisy     Q15 emulator (Daisy Seed config)"
	@echo "  make embedded arm       ARM Q15 synth with FFT analysis"
	@echo "  make embedded arm_restricted  ARM Q15 synth, no analysis"
	@echo "  make cuda               Standalone CUDA build"
	@echo "  make daisy              Cross-compile for Daisy Seed"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make clean              Remove all build artifacts"
	@echo "  make debug              Build with debug symbols"
	@echo "  make info               Show detailed build matrix"
	@echo "  make deps               Install dependencies"
	@echo "  make help               Show this message"
	@echo ""
	@echo "Test Targets:"
	@echo "  make syntax-test        Test all configurations compile"
	@echo "  make convert_segments   Build segment conversion tool"

info:
	@echo ""
	@echo "+--------------------------------------------------------------------------+"
	@echo "|                        SPECTRAL ENGINE BUILD MATRIX                      |"
	@echo "+--------------------------------------------------------------------------+"
	@echo "| Target                  | Analysis | Synth      | GPU   | Sample | Runs  |"
	@echo "+-------------------------+----------+------------+-------+--------+-------+"
	@echo "| desktop (default)       | YES      | CPU float  | YES   | float  | Desk  |"
	@echo "| emulator                | YES      | Q15 emu    | NO    | Q15    | Desk  |"
	@echo "| emulator daisy          | NO       | Q15 emu    | NO    | Q15    | Desk  |"
	@echo "| embedded arm            | YES      | Embedded   | NO    | Q15    | ARM   |"
	@echo "| embedded arm_restricted | NO       | Embedded   | NO    | Q15    | ARM   |"
	@echo "| daisy                   | NO       | (api/)     | NO    | Q15    | ARM   |"
	@echo "+-------------------------+----------+------------+-------+--------+-------+"
	@echo ""
	@echo "+--------------------------------------------------------------------------+"
	@echo "|                         COMPILE FLAGS REFERENCE                          |"
	@echo "+--------------------------------------------------------------------------+"
	@echo "| Flag                        | Description                                |"
	@echo "+-----------------------------+--------------------------------------------+"
	@echo "| SPECTRAL_EMBEDDED           | 1=Q15 fixed-point, 0=float (default)       |"
	@echo "| SPECTRAL_RESTRICTED_MODE    | 1=No FFT/GPU, 0=full features (default)    |"
	@echo "| SPECTRAL_USE_EMBEDDED_SYNTH | 1=Use embedded synth backend               |"
	@echo "| SPECTRAL_COMPACT_SEG        | 1=32-byte segments, 0=64-byte (default)    |"
	@echo "| SPECTRAL_NO_PERF            | 1=Disable perf tracking                    |"
	@echo "| SPECTRAL_MAX_SEGS           | Maximum segment count (0=unlimited)        |"
	@echo "| SPECTRAL_DEBUG_RESTRICTED   | 1=Debug profiling in restricted mode       |"
	@echo "| USE_CUDA                    | 1=Enable CUDA (auto-detected)              |"
	@echo "+-----------------------------+--------------------------------------------+"
	@echo ""
	@echo "Current System: $(UNAME_S)"
ifeq ($(UNAME_S),Darwin)
	@echo "Compiler: $(CC) version $(shell clang --version 2>/dev/null | head -1 | sed 's/.*version //' | cut -d' ' -f1)"
	@echo "Metal: Available"
else
	@echo "Compiler: $(CC) version $(shell gcc -dumpversion 2>/dev/null)"
endif
ifneq ($(NVCC),)
	@echo "CUDA: Available ($(shell nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | cut -d',' -f1))"
else
	@echo "CUDA: Not available"
endif
	@echo ""
	@echo "Minimum compiler requirements: clang 14+ (macOS), gcc 11+ (Linux)"
	@echo "To skip version check: make SKIP_VERSION_CHECK=1"
	@echo ""

# 
#                             TEST TARGETS
# 

.PHONY: syntax-test
syntax-test:
	@echo "Testing all build configurations..."
	@echo ""
	@echo "[1/4] Desktop configuration..."
	$(CC) $(DESKTOP_CFLAGS) -fsyntax-only main.c
	$(CC) $(DESKTOP_CFLAGS) -fsyntax-only spectral_analysis.c
	$(CC) $(DESKTOP_CFLAGS) -fsyntax-only spectral_synth_cpu.c
	@echo "      Desktop: OK"
	@echo ""
	@echo "[2/4] ARM configuration (with analysis)..."
	$(CC) $(ARM_CFLAGS) -fsyntax-only main.c
	$(CC) $(ARM_CFLAGS) -fsyntax-only spectral_analysis.c
	$(CC) $(ARM_CFLAGS) -fsyntax-only spectral_synth_embedded.c
	@echo "      ARM: OK"
	@echo ""
	@echo "[3/4] ARM restricted configuration..."
	$(CC) $(RESTRICTED_CFLAGS) -fsyntax-only main.c
	$(CC) $(RESTRICTED_CFLAGS) -fsyntax-only spectral_synth_embedded.c
	@echo "      ARM restricted: OK"
	@echo ""
	@echo "[4/4] Core modules..."
	$(CC) $(CFLAGS_BASE) -fsyntax-only spectral_common.c
	$(CC) $(CFLAGS_BASE) -fsyntax-only spectral_wavetable.c
	$(CC) $(CFLAGS_BASE) -fsyntax-only spectral_q15.c
	$(CC) $(CFLAGS_BASE) -fsyntax-only spectral_perf.c
	@echo "      Core: OK"
	@echo ""
	@echo "All syntax tests passed!"

# 
#                           SEGMENT CONVERTER
# 

convert_segments: $(BIN_DIR) ../tools/convert_segments.c spectral_common.h
	$(CC) $(CFLAGS_BASE) ../tools/convert_segments.c -o $(BIN_DIR)/convert_segments
	@echo "Built $(BIN_DIR)/convert_segments"

# 
#                            DEPENDENCIES
# 

deps:
ifeq ($(UNAME_S),Darwin)
	@echo "Installing macOS dependencies via Homebrew..."
	@echo ""
	brew install libsndfile libomp
	@echo ""
	@echo "Verifying compiler version..."
	@clang --version | head -1
	@echo ""
	@echo "Dependencies installed. Run 'make info' to verify configuration."
else
	@echo "Installing Linux dependencies..."
	@echo ""
	sudo apt-get update
	sudo apt-get install -y libsndfile1-dev libfftw3-dev libomp-dev gcc-11 g++-11
	@echo ""
	@echo "Setting gcc-11 as default (optional):"
	@echo "  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100"
	@echo "  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100"
	@echo ""
	@gcc --version | head -1
	@echo ""
	@echo "Dependencies installed. Run 'make info' to verify configuration."
endif

# 
#                              PHONY TARGETS
# 

.PHONY: all desktop embedded arm arm_restricted cuda daisy daisy-flash daisy-clean \
        debug clean info deps help syntax-test convert_segments
