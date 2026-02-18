CMAKE ?= cmake
CMAKE_GENERATOR ?=
BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Release
CMAKE_BUILD_CONFIG ?= $(CMAKE_BUILD_TYPE)
CMAKE_CONFIGURE_ARGS ?=

ifneq ($(strip $(CMAKE_GENERATOR)),)
CMAKE_GENERATOR_ARG := -G "$(CMAKE_GENERATOR)"
endif

ifneq ($(strip $(CMAKE_BUILD_TYPE)),)
CMAKE_BUILD_TYPE_ARG := -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
endif

ifneq ($(strip $(CMAKE_GENERATOR)),)
ifneq ($(findstring Multi-Config,$(CMAKE_GENERATOR)),)
CMAKE_BUILD_TYPE_ARG :=
endif
ifneq ($(findstring Visual Studio,$(CMAKE_GENERATOR)),)
CMAKE_BUILD_TYPE_ARG :=
endif
ifneq ($(findstring Xcode,$(CMAKE_GENERATOR)),)
CMAKE_BUILD_TYPE_ARG :=
endif
endif

ifneq ($(strip $(CMAKE_BUILD_CONFIG)),)
CMAKE_BUILD_CONFIG_ARG := --config $(CMAKE_BUILD_CONFIG)
endif

CMAKE_CONFIGURE = $(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_GENERATOR_ARG) \
	$(CMAKE_BUILD_TYPE_ARG) \
	$(CMAKE_CONFIGURE_ARGS)

.PHONY: all configure \
	desktop simulate simulate-daisy simulate_daisy simulate-board simulate_board \
	embedded_arm embedded_arm_float embedded_arm_restricted \
	cuda daisy daisy-flash daisy_flash daisy-clean daisy_clean daisy-example daisy_example daisy-example-flash daisy_example_flash \
	convert_segments log-check log_check syntax-test syntax_test \
	bench bench-cache bench_cache bench-all bench_all \
	info help clean clean-cache clean-output-cache clean_output_cache \
	clean-cmake-cache clean_cmake_cache \
	clean-build-cache clean_build_cache \
	clean-sounds clean-all-sounds clean_all_sounds \
	distclean

all: desktop

configure:
	$(CMAKE_CONFIGURE)

build-%: configure
	$(CMAKE) --build $(BUILD_DIR) --target $* $(CMAKE_BUILD_CONFIG_ARG) --parallel

desktop: build-desktop
simulate: build-simulate
simulate-daisy: build-simulate_daisy
simulate_daisy: simulate-daisy
simulate-board: build-simulate_board
simulate_board: simulate-board

embedded_arm: build-embedded_arm
embedded_arm_float: build-embedded_arm_float
embedded_arm_restricted: build-embedded_arm_restricted

cuda: build-cuda

daisy: build-daisy
daisy-flash: build-daisy_flash
daisy_flash: daisy-flash
daisy-clean: build-daisy_clean
daisy_clean: daisy-clean
daisy-example: build-daisy_example
daisy_example: daisy-example
daisy-example-flash: build-daisy_example_flash
daisy_example_flash: daisy-example-flash

convert_segments: build-convert_segments
log-check: build-log_check
log_check: log-check
syntax-test: build-syntax_test
syntax_test: syntax-test

bench: build-bench
bench-cache: build-bench_cache
bench_cache: bench-cache
bench-all: build-bench_all
bench_all: bench-all

info: build-info

help:
	@echo ""
	@echo "SPECTRAL BUILD"
	@echo ""
	@echo "Configure:"
	@echo "  make configure"
	@echo "  make configure CMAKE_BUILD_TYPE=Debug"
	@echo "  make configure CMAKE_GENERATOR=Ninja"
	@echo "  make desktop CMAKE_BUILD_CONFIG=Debug"
	@echo "  make configure CMAKE_CONFIGURE_ARGS='-DSPECTRAL_USE_CUDA=ON'   # Linux + nvcc"
	@echo "  make configure CMAKE_CONFIGURE_ARGS='-DSPECTRAL_SIMULATION_BOARD=daisy'"
	@echo "  build flags source: spectral_engine/cmake/options.cmake"
	@echo ""
	@echo "Primary Targets:"
	@echo "  desktop"
	@echo "  simulate | simulate-daisy | simulate-board"
	@echo "  embedded_arm | embedded_arm_float | embedded_arm_restricted"
	@echo "  cuda (desktop alias; enables CUDA backend when configured)"
	@echo "  daisy | daisy-flash | daisy-clean"
	@echo "  daisy-example | daisy-example-flash"
	@echo "  convert_segments"
	@echo "  log-check | syntax-test"
	@echo "  bench | bench-cache | bench-all"
	@echo "  info"
	@echo "  clean | clean-cache | clean-cmake-cache | clean-output | clean-sounds | distclean"
	@echo "  aliases: clean-output-cache clean-build-cache clean-all-sounds"
	@echo ""
	@echo "Build Matrix:"
	@echo "  desktop                 analysis=YES synth=CPU-float   gpu=YES run=desktop"
	@echo "  simulate                analysis=YES synth=Q15-sim     gpu=NO  run=desktop"
	@echo "  simulate_daisy          analysis=NO  synth=Q15-sim     gpu=NO  run=desktop"
	@echo "  embedded_arm            analysis=YES synth=embedded    gpu=NO  run=host-built"
	@echo "  embedded_arm_float      analysis=YES synth=embedded-fp gpu=NO  run=host-built"
	@echo "  embedded_arm_restricted analysis=NO  synth=embedded    gpu=NO  run=host-built"
	@echo "  cuda                    alias for desktop (CUDA backend when enabled)"
	@echo "  daisy                   analysis=NO  synth=api/daisy   gpu=NO  run=arm"
	@echo "  daisy_example           analysis=NO  synth=examples    gpu=NO  run=arm"
	@echo ""
	@echo "Artifacts:"
	@echo "  CMake metadata:  $(BUILD_DIR)/"
	@echo "  Binaries:        $(BUILD_DIR)/bin/"
	@echo "  Daisy binaries:  $(BUILD_DIR)/bin/daisy/"
	@echo "  Runtime output:  output/"
	@echo ""
	@echo "Examples:"
	@echo "  make desktop"
	@echo "  make cuda CMAKE_CONFIGURE_ARGS='-DSPECTRAL_USE_CUDA=ON'"
	@echo "  make simulate-daisy"
	@echo "  make daisy CMAKE_CONFIGURE_ARGS='-DSPECTRAL_DAISY_LIBDAISY_DIR=/path/to/libDaisy -DSPECTRAL_DAISY_DAISYSP_DIR=/path/to/DaisySP'"
	@echo ""

clean:
	rm -rf $(BUILD_DIR)

clean-cache:
	rm -rf output/cache output/pgo

clean-output-cache: clean-cache
clean_output_cache: clean-output-cache

clean-cmake-cache:
	find $(BUILD_DIR) -type d -name CMakeFiles -prune -exec rm -rf {} + 2>/dev/null || true
	find $(BUILD_DIR) -type f \( \
		-name CMakeCache.txt -o -name cmake_install.cmake -o \
		-name CTestTestfile.cmake -o \
		-name compile_commands.json -o \
		-name spectral_log_check.cmake -o \
		-name Makefile -o -name build.ninja -o -name rules.ninja -o \
		-name .ninja_deps -o -name .ninja_log \
	\) -delete 2>/dev/null || true

clean-build-cache: clean-cmake-cache
clean_build_cache: clean-build-cache
clean_cmake_cache: clean-cmake-cache

clean-output:
	find output -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true

clean-sounds:
	find output $(BUILD_DIR) -type f \( \
		-name '*.wav' -o -name '*.aif' -o -name '*.aiff' -o \
		-name '*.flac' -o -name '*.mp3' -o -name '*.ogg' \
	\) -delete 2>/dev/null || true
	rm -f out*.wav

clean-all-sounds: clean-sounds
clean_all_sounds: clean-all-sounds

distclean: clean
	rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake
	rm -rf spectral_engine/CMakeFiles spectral_engine/Makefile spectral_engine/cmake_install.cmake
	rm -rf api/daisy_seed/CMakeFiles api/daisy_seed/Makefile api/daisy_seed/cmake_install.cmake
	rm -rf examples/daisy_seed/CMakeFiles examples/daisy_seed/Makefile examples/daisy_seed/cmake_install.cmake
