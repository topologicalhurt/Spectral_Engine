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
#   make embedded_arm            Cross-compile: ARM Q15 with FFT analysis
#   make embedded_arm_float       Cross-compile: ARM float synth (FPU) with analysis
#   make embedded_arm_restricted  Cross-compile: ARM Q15, NO analysis
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
# embedded_arm              | YES      | spectral_synth_embedded | NO | Q15        | ARM
# embedded_arm_float        | YES      | spectral_synth_embedded | NO | Q15->float | ARM
# embedded_arm_restricted   | NO       | spectral_synth_embedded | NO | Q15        | ARM
# daisy                     | NO       | (via api/)         | NO   | Q15         | ARM
#
# COMPILE FLAGS REFERENCE:
# ------------------------
# Flag                      | Description
# --------------------------|--------------------------------------------------
# SPECTRAL_EMBEDDED         | 1=Q15 fixed-point, 0=float (default)
# SPECTRAL_EMBEDDED_FLOAT   | 1=Use FPU for synth on embedded (requires FPU)
# SPECTRAL_RESTRICTED_MODE  | 1=No FFT analysis/GPU, 0=full features (default)
# SPECTRAL_EMBEDDED_EMULATION | 1=Desktop emulating Q15 path (testing)
# SPECTRAL_USE_EMBEDDED_SYNTH | 1=Use embedded synth instead of CPU synth
# SPECTRAL_COMPACT_SEG      | 1=32-byte segments, 0=64-byte (default)
# SPECTRAL_NO_PERF          | 1=Disable perf tracking, 0=enabled (default)
# SPECTRAL_MAX_SEGS         | Maximum segment count (0=unlimited)
# SPECTRAL_DEBUG_RESTRICTED | 1=Enable debug profiling in restricted mode
# SPECTRAL_OPT_LEVEL        | Optimization level for quality/speed tradeoffs:
#                           |   0=Conservative (saturating math, interpolation)
#                           |   1=Balanced (non-saturating math) [default]
#                           |   2=Aggressive (level 1 + no LUT interpolation)
#                           |   3=Reserved for future use
# USE_CUDA                  | 1=Enable CUDA backend (auto-detected)
#
# OUTPUT DIRECTORY:
#   All binaries placed in ../bin/ (project root)
#
# 

# 
#                              CONFIGURATION
# 

UNAME_S := $(shell uname -s)
ARCH    := $(shell uname -m)
NVCC    := $(shell command -v nvcc 2>/dev/null)

NPROCS := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
MAKEFLAGS += -j$(NPROCS)

CC ?= $(if $(filter Darwin,$(UNAME_S)),clang,gcc)

HAS_METAL := 0
HAS_CUDA  := 0
ifeq ($(UNAME_S),Darwin)
    HAS_METAL := 1
    GPU_BACKEND := metal
else ifneq ($(NVCC),)
    HAS_CUDA := 1
    GPU_BACKEND := cuda
else
    GPU_BACKEND := native
endif

USE_CUDA ?= $(HAS_CUDA)
ifeq ($(HAS_CUDA),0)
    override USE_CUDA := 0
endif

BIN_DIR := ../bin

OBJ_DIR           := $(BIN_DIR)/obj_$(ARCH)_$(GPU_BACKEND)_desktop
OBJ_EMU_DIR       := $(BIN_DIR)/obj_$(ARCH)_native_emulator
OBJ_EMU_DAISY_DIR := $(BIN_DIR)/obj_$(ARCH)_native_emulator_daisy
OBJ_EMBEDDED_ARM_FLOAT_DIR := $(BIN_DIR)/obj_embedded_arm32_float
OBJ_EMBEDDED_ARM_DIR       := $(BIN_DIR)/obj_embedded_arm32
OBJ_EMBEDDED_ARM_RES_DIR   := $(BIN_DIR)/obj_embedded_arm32_restricted

TARGET_DESKTOP              := spectral_$(ARCH)_$(GPU_BACKEND)_desktop
TARGET_EMULATOR             := spectral_$(ARCH)_native_emulator
TARGET_EMULATOR_DAISY       := spectral_$(ARCH)_native_emulator_daisy
TARGET_EMBEDDED_ARM         := spectral_embedded_arm32
TARGET_EMBEDDED_ARM_FLOAT   := spectral_embedded_arm32_float
TARGET_EMBEDDED_ARM_RES     := spectral_embedded_arm32_restricted
TARGET_CUDA           		:= spectral_$(ARCH)_cuda_standalone

# 
#                      COMPILER VERSION CHECK
# 
# Override: make SKIP_VERSION_CHECK=1

MIN_CLANG_VERSION := 14.0.0
MIN_GCC_VERSION   := 11.0.0

CC_IS_CLANG := $(findstring clang,$(shell $(CC) --version 2>/dev/null | head -1))
ifneq ($(CC_IS_CLANG),)
    CC_VERSION := $(shell $(CC) --version 2>/dev/null | grep -oE 'version [0-9]+\.[0-9]+' | head -1 | cut -d' ' -f2)
    MIN_VERSION := $(MIN_CLANG_VERSION)
else
    CC_VERSION := $(shell $(CC) -dumpversion 2>/dev/null)
    MIN_VERSION := $(MIN_GCC_VERSION)
endif

CC_VERSION_OK := $(shell printf '%s\n%s' "$(MIN_VERSION)" "$(CC_VERSION)" | sort -V | head -1 | grep -qF "$(MIN_VERSION)" && echo 1 || echo 0)

# Skip version check for non-build targets
NON_BUILD_TARGETS := clean help info deps update_submodules
IS_BUILD_TARGET := $(if $(filter-out $(NON_BUILD_TARGETS),$(MAKECMDGOALS)),1,$(if $(MAKECMDGOALS),,1))

ifndef SKIP_VERSION_CHECK
ifeq ($(CC_VERSION_OK),0)
ifeq ($(IS_BUILD_TARGET),1)
$(error $(CC) version $(CC_VERSION) is too old. Minimum: $(MIN_VERSION). Skip: make SKIP_VERSION_CHECK=1)
endif
endif
endif

# 
#                            SOURCE FILE GROUPS
# 

CORE_DIR := core
VPATH    := $(CORE_DIR)
HEADERS  := $(wildcard *.h) $(wildcard $(CORE_DIR)/*.h)

SRCS_CORE := $(wildcard $(CORE_DIR)/*.c)
SRCS_MONITORING := spectral_perf.c

SRCS_ANALYSIS       := spectral_analysis.c
SRCS_SYNTH_CPU      := spectral_synth_cpu.c
SRCS_SYNTH_EMBEDDED := spectral_synth_arm32.c $(CORE_DIR)/spectral_q15.c
SRCS_SYNTH_EMULATOR := spectral_synth_emulator.c

SRCS_METAL := spectral_synth_metal.m
SRCS_CUDA  := spectral_synth_cuda.cu

# 
#                         PLATFORM-SPECIFIC CONFIG
# 

AUDIO_OPT_FLAGS := -ffast-math -fno-signed-zeros -fno-trapping-math \
                   -fassociative-math -freciprocal-math -ffp-contract=fast \
                   -funroll-loops -ftree-vectorize -fomit-frame-pointer

ifeq ($(UNAME_S),Darwin)
    # macOS configuration
    # Note: CC already set in CONFIGURATION section
    
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
                   -I. -I$(CORE_DIR) -I../third_party/simde \
                   -I$(OMP_PREFIX)/include -I$(SNDFILE_PREFIX)/include \
                   -Xpreprocessor -fopenmp
    
    OBJCFLAGS := $(CFLAGS_BASE) -fobjc-arc
    
    # LTO requires passing flags to linker
    LDFLAGS_BASE := -flto=thin -lm -L$(OMP_PREFIX)/lib -lomp -L$(SNDFILE_PREFIX)/lib -lsndfile
    LDFLAGS_GPU  := $(LDFLAGS_BASE) -framework Accelerate -framework Metal -framework Foundation
    LDFLAGS_CPU  := $(LDFLAGS_BASE) -framework Accelerate

else
    # Linux configuration
    # Note: CC already set in CONFIGURATION section
    
    # Base flags with audio DSP optimizations
    # -flto=auto: Full LTO with automatic parallelism
    # -floop-nest-optimize: Optimize nested loops (common in synthesis)
    # -fgraphite-identity: Enable Graphite loop optimizer
    # -fipa-pta: Interprocedural pointer analysis for better aliasing
    CFLAGS_BASE  := -std=c11 -O3 -march=native -mtune=native $(AUDIO_OPT_FLAGS) \
                    -flto=auto -floop-nest-optimize -fgraphite-identity -fipa-pta \
                    -Wall -Wextra -Wno-unknown-pragmas \
                    -I. -I$(CORE_DIR) -I../third_party/simde -fopenmp
    
    # LTO requires passing flags to linker
    LDFLAGS_BASE := -flto=auto -lm -lsndfile -lfftw3f -fopenmp
    LDFLAGS_GPU  := $(LDFLAGS_BASE)
    LDFLAGS_CPU  := $(LDFLAGS_BASE)
    
    # CUDA support on Linux
	ifeq ($(USE_CUDA),1)
        CFLAGS_BASE  += -D USE_CUDA
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
DESKTOP_SRCS_C := main.c $(SRCS_CLI) $(SRCS_ANALYSIS) $(SRCS_SYNTH_CPU) $(SRCS_CORE) $(SRCS_MONITORING)

# Convert source paths to flat object paths (core/foo.c -> foo.o)
define src_to_obj
$(addprefix $(1)/,$(notdir $(2:.c=.o)))
endef

ifeq ($(HAS_METAL),1)
    DESKTOP_SRCS_OBJC := $(SRCS_METAL)
    DESKTOP_OBJS := $(call src_to_obj,$(OBJ_DIR),$(DESKTOP_SRCS_C)) \
                    $(addprefix $(OBJ_DIR)/,$(DESKTOP_SRCS_OBJC:.m=.o))
    DESKTOP_LDFLAGS := $(LDFLAGS_GPU)
else
    DESKTOP_OBJS := $(call src_to_obj,$(OBJ_DIR),$(DESKTOP_SRCS_C))
    ifeq ($(USE_CUDA),1)
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
ifeq ($(HAS_METAL),1)
	@echo "  - Metal GPU backend"
endif
ifeq ($(USE_CUDA),1)
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

EMULATOR_CFLAGS := $(CFLAGS_BASE) -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_USE_EMBEDDED_SYNTH=1 \
                   -DSPECTRAL_EMBEDDED_EMULATION=1
EMULATOR_SRCS_C := main.c $(SRCS_CLI) $(SRCS_ANALYSIS) $(SRCS_SYNTH_CPU) $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) \
                   $(SRCS_CORE) $(SRCS_MONITORING)
EMULATOR_OBJS   := $(call src_to_obj,$(OBJ_EMU_DIR),$(EMULATOR_SRCS_C))

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
EMULATOR_DAISY_CFLAGS := $(CFLAGS_BASE) -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_USE_EMBEDDED_SYNTH=1 \
                         -DSPECTRAL_EMBEDDED_EMULATION=1 -DSPECTRAL_RESTRICTED_MODE=1 \
                         -DEMULATOR_TARGET_DAISY=1
EMULATOR_DAISY_SRCS_C := main.c $(SRCS_CLI) $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) $(SRCS_CORE) $(SRCS_MONITORING)
EMULATOR_DAISY_OBJS   := $(call src_to_obj,$(OBJ_EMU_DAISY_DIR),$(EMULATOR_DAISY_SRCS_C))

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
$(OBJ_EMU_DIR)/%.o: %.c | $(OBJ_EMU_DIR)
	$(CC) $(EMULATOR_CFLAGS) $(DEPFLAGS) -c $< -o $@

# Emulator object files (Daisy)
$(OBJ_EMU_DAISY_DIR)/%.o: %.c | $(OBJ_EMU_DAISY_DIR)
	$(CC) $(EMULATOR_DAISY_CFLAGS) $(DEPFLAGS) -c $< -o $@

# 
#                          TARGET: embedded arm
# 
# Cross-compile ARM Q15 synthesis with FFT analysis:
#   - FFT analysis enabled
#   - Q15 fixed-point synthesis (spectral_synth_arm32.c)
#   - NO GPU backends
#   - Binary targets ARM architecture (won't run on desktop)
#
# NOTE: This is for CROSS-COMPILATION. Use 'make emulator' to test
# the Q15 path on desktop.

# Common flags shared by all embedded targets
EMBEDDED_COMMON_FLAGS := -ffast-math -funsafe-math-optimizations \
                         -fdata-sections -ffunction-sections \
                         -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_USE_EMBEDDED_SYNTH=1

ARM_EMBEDDED_FLAGS     := $(EMBEDDED_COMMON_FLAGS)

EMBEDDED_ARM_CFLAGS := $(CFLAGS_BASE) $(ARM_EMBEDDED_FLAGS)
EMBEDDED_ARM_SRCS_C := main.c $(SRCS_ANALYSIS) $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) $(SRCS_CORE) $(SRCS_MONITORING)
EMBEDDED_ARM_OBJS   := $(call src_to_obj,$(OBJ_EMBEDDED_ARM_DIR),$(EMBEDDED_ARM_SRCS_C))

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

# Aggressive optimization flags for restricted embedded ARM targets
# These mirror the flags used in api/daisy_seed/Makefile for Cortex-M7
EMBEDDED_OPT_FLAGS := $(EMBEDDED_COMMON_FLAGS) -fno-exceptions -fno-threadsafe-statics \
                      -DSPECTRAL_RESTRICTED_MODE=1 -DSPECTRAL_NO_PERF=1

EMBEDDED_ARM_RES_CFLAGS := $(CFLAGS_BASE) $(EMBEDDED_OPT_FLAGS)
EMBEDDED_ARM_RES_SRCS_C := main.c $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) $(SRCS_CORE)
EMBEDDED_ARM_RES_OBJS   := $(call src_to_obj,$(OBJ_EMBEDDED_ARM_RES_DIR),$(EMBEDDED_ARM_RES_SRCS_C))

.PHONY: embedded_arm
embedded_arm: $(BIN_DIR)/$(TARGET_EMBEDDED_ARM)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_EMBEDDED_ARM)"
	@echo "================================================"
	@echo "Embedded ARM Cross-Compile (with analysis):"
	@echo "  - FFT analysis enabled"
	@echo "  - Q15 fixed-point synthesis"
	@echo "  - Target: ARM Cortex-M"
	@echo ""
	@echo "NOTE: Binary won't run on this machine."
	@echo "      Use 'make emulator' to test Q15 path on desktop."
	@echo "================================================"

$(BIN_DIR)/$(TARGET_EMBEDDED_ARM): $(EMBEDDED_ARM_OBJS) | $(BIN_DIR)
	$(CC) $(EMBEDDED_ARM_OBJS) -o $@ $(LDFLAGS_CPU)

# ARM float target (FPU synthesis, with analysis)
ARM_FLOAT_FLAGS := $(EMBEDDED_COMMON_FLAGS) -DSPECTRAL_EMBEDDED_FLOAT=1

EMBEDDED_ARM_FLOAT_CFLAGS := $(CFLAGS_BASE) $(ARM_FLOAT_FLAGS)
EMBEDDED_ARM_FLOAT_SRCS_C := main.c $(SRCS_ANALYSIS) $(SRCS_SYNTH_EMBEDDED) $(SRCS_SYNTH_EMULATOR) $(SRCS_CORE) $(SRCS_MONITORING)
EMBEDDED_ARM_FLOAT_OBJS   := $(call src_to_obj,$(OBJ_EMBEDDED_ARM_FLOAT_DIR),$(EMBEDDED_ARM_FLOAT_SRCS_C))

$(OBJ_EMBEDDED_ARM_FLOAT_DIR):
	mkdir -p $@

$(OBJ_EMBEDDED_ARM_FLOAT_DIR)/%.o: %.c | $(OBJ_EMBEDDED_ARM_FLOAT_DIR)
	$(CC) $(EMBEDDED_ARM_FLOAT_CFLAGS) $(DEPFLAGS) -c $< -o $@

.PHONY: embedded_arm_float
embedded_arm_float: $(BIN_DIR)/$(TARGET_EMBEDDED_ARM_FLOAT)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_EMBEDDED_ARM_FLOAT)"
	@echo "================================================"
	@echo "Embedded ARM Cross-Compile Float (with analysis):"
	@echo "  - FFT analysis enabled"
	@echo "  - Float FPU synthesis (SPECTRAL_EMBEDDED_FLOAT=1)"
	@echo "  - Q15 storage, float compute"
	@echo "  - Target: ARM with VFPv4 FPU"
	@echo ""
	@echo "NOTE: Binary won't run on this machine."
	@echo "      Requires ARM with hardware FPU."
	@echo "================================================"

$(BIN_DIR)/$(TARGET_EMBEDDED_ARM_FLOAT): $(EMBEDDED_ARM_FLOAT_OBJS) | $(BIN_DIR)
	$(CC) $(EMBEDDED_ARM_FLOAT_OBJS) -o $@ $(LDFLAGS_CPU)

.PHONY: embedded_arm_restricted
embedded_arm_restricted: $(BIN_DIR)/$(TARGET_EMBEDDED_ARM_RES)
	@echo ""
	@echo "================================================"
	@echo "Built: $(BIN_DIR)/$(TARGET_EMBEDDED_ARM_RES)"
	@echo "================================================"
	@echo "Embedded ARM Cross-Compile Restricted (Daisy Seed):"
	@echo "  - NO FFT analysis (load pre-analyzed segments)"
	@echo "  - Q15 fixed-point synthesis"
	@echo "  - Target: ARM architecture"
	@echo ""
	@echo "NOTE: Binary won't run on this machine."
	@echo "      Use 'make emulator' to test Q15 path on desktop."
	@echo "================================================"

$(BIN_DIR)/$(TARGET_EMBEDDED_ARM_RES): $(EMBEDDED_ARM_RES_OBJS) | $(BIN_DIR)
	$(CC) $(EMBEDDED_ARM_RES_OBJS) -o $@ $(LDFLAGS_CPU)

# 
#                             TARGET: cuda
# 
# Standalone CUDA build (Linux only)

ifeq ($(filter 01,$(HAS_CUDA)$(USE_CUDA)),01)
    CUDA_ERROR := 1
endif

cuda: $(BIN_DIR)
ifeq ($(CUDA_ERROR),1)
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
# When used as modifier, daisy-build is skipped

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

# Automatic dependency generation — rebuilds only what changed
DEPFLAGS = -MMD -MP -MF $(@:.o=.d)

# Create directories
$(BIN_DIR) $(OBJ_DIR) $(OBJ_EMBEDDED_ARM_DIR) $(OBJ_EMBEDDED_ARM_RES_DIR) $(OBJ_EMU_DIR) $(OBJ_EMU_DAISY_DIR):
	@mkdir -p $@

# Desktop object files
$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	$(CC) $(DESKTOP_CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.m | $(OBJ_DIR)
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/spectral_synth_cuda.o: spectral_synth_cuda.cu $(HEADERS) | $(OBJ_DIR)
ifeq ($(USE_CUDA),1)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

# Embedded ARM object files (with analysis)
$(OBJ_EMBEDDED_ARM_DIR)/%.o: %.c | $(OBJ_EMBEDDED_ARM_DIR)
	$(CC) $(EMBEDDED_ARM_CFLAGS) $(DEPFLAGS) -c $< -o $@

# Embedded ARM restricted object files (no analysis)
$(OBJ_EMBEDDED_ARM_RES_DIR)/%.o: %.c | $(OBJ_EMBEDDED_ARM_RES_DIR)
	$(CC) $(EMBEDDED_ARM_RES_CFLAGS) $(DEPFLAGS) -c $< -o $@

# 
#                              UTILITIES
# 

.PHONY: debug clean info deps help

debug: DESKTOP_CFLAGS += -g -O0 -DDEBUG
debug: EMBEDDED_ARM_CFLAGS += -g -O0 -DDEBUG
debug: EMBEDDED_ARM_RES_CFLAGS += -g -O0 -DDEBUG
debug: clean all

clean:
	rm -rf $(BIN_DIR) out_c.wav segments.bin
	@echo "Cleaned all build artifacts."

#
#                               HELP & INFO
#

help:
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════════════╗"
	@echo "║                         SPECTRAL ENGINE BUILD SYSTEM                      ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build Targets:"
	@printf "  %-26s %s\n" "desktop" "Full desktop build (Metal/CUDA, FFT, float) [default]"
	@printf "  %-26s %s\n" "emulator" "Q15 embedded emulation on desktop"
	@printf "  %-26s %s\n" "emulator daisy" "Daisy Seed emulation (STM32H750 @ 480MHz)"
	@printf "  %-26s %s\n" "embedded_arm" "Cross-compile embedded ARM Q15 + FFT"
	@printf "  %-26s %s\n" "embedded_arm_float" "Cross-compile embedded ARM float FPU + FFT"
	@printf "  %-26s %s\n" "embedded_arm_restricted" "Cross-compile embedded ARM Q15, no analysis"
	@printf "  %-26s %s\n" "cuda" "Standalone CUDA build (Linux only)"
	@printf "  %-26s %s\n" "daisy" "Cross-compile for Daisy Seed hardware"
	@echo ""
	@echo "Utility Targets:"
	@printf "  %-26s %s\n" "clean" "Remove all build artifacts"
	@printf "  %-26s %s\n" "debug" "Build with debug symbols (-g -O0)"
	@printf "  %-26s %s\n" "deps" "Install dependencies + pull subtrees"
	@printf "  %-26s %s\n" "update_submodules" "Pull specific git subtrees"
	@printf "  %-26s %s\n" "convert_segments" "Build segment conversion tool"
	@printf "  %-26s %s\n" "syntax-test" "Verify all configurations compile"
	@printf "  %-26s %s\n" "daisy-flash" "Flash firmware to Daisy Seed"
	@printf "  %-26s %s\n" "info" "Show build matrix and system info"
	@printf "  %-26s %s\n" "help" "Show this help message"
	@echo ""
	@echo "Current binary: $(TARGET_DESKTOP)"
	@echo ""
	@echo "Examples:"
	@echo "  make                       Build desktop version"
	@echo "  make emulator daisy        Build Daisy Seed emulator"
	@echo "  make embedded_arm_float    Cross-compile for embedded ARM FPU"
	@echo ""

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
	@echo "| embedded_arm            | YES      | Embedded   | NO    | Q15    | ARM   |"
	@echo "| embedded_arm_float      | YES      | Embedded   | NO    | float  | ARM   |"
	@echo "| embedded_arm_restricted | NO       | Embedded   | NO    | Q15    | ARM   |"
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
	@echo "Source structure:"
	@echo "  Headers: $(words $(HEADERS)) files (auto-discovered)"
	@echo "  Core sources: $(words $(SRCS_CORE)) files in $(CORE_DIR)/"
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
	@echo "[2/4] Embedded ARM configuration (with analysis)..."
	$(CC) $(EMBEDDED_ARM_CFLAGS) -fsyntax-only main.c
	$(CC) $(EMBEDDED_ARM_CFLAGS) -fsyntax-only spectral_analysis.c
	$(CC) $(EMBEDDED_ARM_CFLAGS) -fsyntax-only spectral_synth_arm32.c
	@echo "      Embedded ARM: OK"
	@echo ""
	@echo "[3/4] Embedded ARM restricted configuration..."
	$(CC) $(EMBEDDED_ARM_RES_CFLAGS) -fsyntax-only main.c
	$(CC) $(EMBEDDED_ARM_RES_CFLAGS) -fsyntax-only spectral_synth_arm32.c
	@echo "      Embedded ARM restricted: OK"
	@echo ""
	@echo "[4/4] Core modules..."
	$(CC) $(CFLAGS_BASE) -fsyntax-only $(CORE_DIR)/spectral_common.c
	$(CC) $(CFLAGS_BASE) -fsyntax-only $(CORE_DIR)/spectral_wavetable.c
	$(CC) $(CFLAGS_BASE) -fsyntax-only $(CORE_DIR)/spectral_q15.c
	$(CC) $(CFLAGS_BASE) -fsyntax-only $(CORE_DIR)/spectral_perf.c
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

# Third party libraries update (managed via git subtrees)
# Usage: make update_submodules SUBTREE_ARGS="lib/a lib/b"
#        make deps  (pulls all subtrees)
SUBTREE_ARGS ?=
update_submodules:
	@if [ -z "$(SUBTREE_ARGS)" ]; then \
		echo "Usage: make update_submodules SUBTREE_ARGS=\"lib/a lib/b\""; \
		echo ""; \
		echo "To update ALL subtrees, run: make deps update_submodules"; \
		echo "Or directly: ../subtrees/update.sh pull"; \
		exit 1; \
	else \
		bash ../subtrees/update.sh pull $(SUBTREE_ARGS); \
	fi

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
ifneq ($(filter update_submodules,$(MAKECMDGOALS)),)
	@echo ""
	@echo "Pulling third-party subtrees..."
	@bash ../subtrees/update.sh pull
endif

# 
#                            OUT OF SOURCE BUILDS
# 


# TODO
# SUBTREES_TO_MAKE := ../subtrees/LLAC/Src/Allocator/Core/libs/

# build_subtrees: 
# 	${foreach subtree, $(SUBTREES_TO_MAKE), $(MAKE) -C $(subtree);}

# 
#                              PHONY TARGETS
# 

.PHONY: all desktop embedded_arm embedded_arm_float embedded_arm_restricted cuda daisy daisy-flash daisy-clean \
        debug clean info deps help syntax-test convert_segments update_submodules build_subtrees

#
#                         AUTO-DEPENDENCY INCLUDES
#
# Generated .d files track header dependencies per object file.
# On first build (no .d files), the pattern rules handle everything.
# On incremental builds, only changed headers trigger recompilation.

-include $(wildcard $(OBJ_DIR)/*.d)
-include $(wildcard $(OBJ_EMU_DIR)/*.d)
-include $(wildcard $(OBJ_EMU_DAISY_DIR)/*.d)
-include $(wildcard $(OBJ_EMBEDDED_ARM_DIR)/*.d)
-include $(wildcard $(OBJ_EMBEDDED_ARM_FLOAT_DIR)/*.d)
-include $(wildcard $(OBJ_EMBEDDED_ARM_RES_DIR)/*.d)
