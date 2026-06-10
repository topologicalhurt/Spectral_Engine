#!/usr/bin/env bash
# Perf-model Layer 1: exact dynamic instruction/memory counts of the REAL
# ARM kernel under qemu-system-arm mps2-an500 (M7_PERF_MODEL_PLAN P2).
# Builds the freestanding runner ELF, the TCG counting plugin, derives symbol
# ranges with nm, runs the deterministic workload, prints the counts table.
#
# Usage: tools/perf_model/run_qemu_counts.sh [out_dir]   (default build/perf_model)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$REPO_ROOT/build/perf_model}"
TOOL_DIR="$REPO_ROOT/tools/perf_model"
QEMU_DIR="$TOOL_DIR/qemu"
SRC_ARM="$REPO_ROOT/spectral_engine/synth/backends/arm"
QEMU_INC="$(dirname "$(dirname "$(readlink -f "$(which qemu-system-arm)")")")/include"
[ -f "$QEMU_INC/qemu-plugin.h" ] || QEMU_INC="/opt/homebrew/include"

CFLAGS=(-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -O3 -ffreestanding
        -isystem "$TOOL_DIR/fs_include"
        -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_ARM_M7=1 -DSPECTRAL_HAS_DUAL_MAC=1
        -I"$REPO_ROOT/spectral_engine" -I"$REPO_ROOT/spectral_engine/core"
        -I"$REPO_ROOT/spectral_engine/synth" -I"$REPO_ROOT/spectral_engine/synth/math"
        -I"$REPO_ROOT/spectral_engine/synth/api" -I"$REPO_ROOT/spectral_engine/runtime"
        -I"$REPO_ROOT/spectral_engine/analysis" -I"$SRC_ARM")

mkdir -p "$OUT"

# --- runner ELF (real kernel TUs + freestanding startup/workload)
GC=(-ffunction-sections -fdata-sections)
arm-none-eabi-gcc "${CFLAGS[@]}" "${GC[@]}" -fno-builtin -fno-tree-loop-distribute-patterns \
    -c "$QEMU_DIR/startup.c" -o "$OUT/startup.o"
arm-none-eabi-gcc "${CFLAGS[@]}" "${GC[@]}" -fno-builtin \
    -c "$QEMU_DIR/qemu_main.c" -o "$OUT/qemu_main.o"
arm-none-eabi-gcc "${CFLAGS[@]}" "${GC[@]}" -c "$SRC_ARM/spectral_synth_arm32.c" -o "$OUT/arm32.o"
arm-none-eabi-gcc "${CFLAGS[@]}" "${GC[@]}" -c "$REPO_ROOT/spectral_engine/synth/math/spectral_q15.c" -o "$OUT/q15.o"
arm-none-eabi-gcc "${CFLAGS[@]}" "${GC[@]}" -c "$REPO_ROOT/spectral_engine/core/spectral_lut.c" -o "$OUT/lut.o"
arm-none-eabi-gcc -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
    -nostdlib -T "$QEMU_DIR/mps2_an500.ld" -Wl,--gc-sections \
    "$OUT/startup.o" "$OUT/qemu_main.o" "$OUT/arm32.o" "$OUT/q15.o" "$OUT/lut.o" \
    -lgcc -o "$OUT/qemu_counts.elf"

# --- symbol ranges -> plugin args (function symbols, [start, start+size))
RANGE_ARGS=""
while read -r addr size _type name; do
    RANGE_ARGS+=",range=$name:0x$addr:0x$(printf '%x' $((0x$addr + 0x$size)))"
done < <(arm-none-eabi-nm -S "$OUT/qemu_counts.elf" \
         | grep -E " [Tt] (spectral_arm32_process|spectral_arm32_load|spectral_lut_init_sine|memcpy|memset|main)$")

# --- TCG plugin (host dylib)
HOST_CC="${HOST_CC:-cc}"
GLIB_FLAGS="$(pkg-config --cflags glib-2.0)"
PLUGIN_LDFLAGS=""
[ "$(uname)" = "Darwin" ] && PLUGIN_LDFLAGS="-Wl,-undefined,dynamic_lookup"
$HOST_CC -O2 -shared -fPIC $GLIB_FLAGS -I"$QEMU_INC" $PLUGIN_LDFLAGS \
    "$QEMU_DIR/spectral_counts.c" -o "$OUT/libspectral_counts.so" \
    $(pkg-config --libs glib-2.0)

# --- run (deterministic: single vcpu, no interrupts, semihosted exit)
qemu-system-arm -M mps2-an500 -semihosting -display none -serial none -monitor none \
    -plugin "$OUT/libspectral_counts.so${RANGE_ARGS},out=$OUT/qemu_counts.txt" \
    -kernel "$OUT/qemu_counts.elf"

echo
cat "$OUT/qemu_counts.txt"
