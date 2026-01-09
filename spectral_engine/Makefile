# Spectral Engine Makefile

TARGET = spectral
SRCS_C = main.c spectral_analysis.c spectral_synth_cpu.c
SRCS_OBJC = spectral_synth_metal.m
HEADERS = spectral_common.h spectral_analysis.h spectral_synth.h
OBJS_C = $(SRCS_C:.c=.o)
OBJS_OBJC = $(SRCS_OBJC:.m=.o)

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    CC = clang
    OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
    SNDFILE_PREFIX := $(shell brew --prefix libsndfile 2>/dev/null || echo /opt/homebrew/opt/libsndfile)
    
    CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -Wno-unknown-pragmas
    CFLAGS += -I$(OMP_PREFIX)/include -I$(SNDFILE_PREFIX)/include -Xpreprocessor -fopenmp
    OBJCFLAGS = $(CFLAGS) -fobjc-arc
    
    LDFLAGS = -lm -L$(OMP_PREFIX)/lib -lomp -L$(SNDFILE_PREFIX)/lib -lsndfile
    LDFLAGS += -framework Accelerate -framework Metal -framework Foundation
    
    OBJS = $(OBJS_C) $(OBJS_OBJC)
else
    CC = gcc
    CFLAGS = -std=c11 -O3 -march=native -ffast-math -Wall -Wextra -Wno-unknown-pragmas -fopenmp
    LDFLAGS = -lm -lsndfile -lfftw3 -fopenmp
    OBJS = $(OBJS_C)
endif

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "Built $(TARGET)"

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.m $(HEADERS)
	$(CC) $(OBJCFLAGS) -c $< -o $@

# ==========================================
#              Utilities
# ==========================================

debug: CFLAGS += -g -O0
debug: OBJCFLAGS += -g -O0
debug: clean $(TARGET)

clean:
	rm -f $(OBJS_C) $(OBJS_OBJC) $(TARGET) spectral_cuda out_c.wav segments.bin
	@echo "Cleaned."

info:
	@echo "================================================"
	@echo "Spectral Engine (Modular)"
	@echo "================================================"
	@echo "System: $(UNAME_S) | Compiler: $(CC)"
	@echo "Sources: $(SRCS_C)"
ifeq ($(UNAME_S),Darwin)
	@echo "         $(SRCS_OBJC)"
endif
	@echo "================================================"
	@echo "Backends: 0=auto 1=cpu 2=metal 3=cuda 4=export"
	@echo "================================================"

deps:
ifeq ($(UNAME_S),Darwin)
	brew install libsndfile libomp
else
	sudo apt-get install -y libsndfile1-dev libfftw3-dev libomp-dev
endif

.PHONY: all clean debug info deps
