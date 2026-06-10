#!/usr/bin/env bash
# Perf-model Layer 0 codegen census + Layer 2 llvm-mca pipeline analysis
# (M7_PERF_MODEL_PLAN P1). Compiles the REAL ARM backend TU with the Daisy
# production flags, reports the DSP/MAC instruction census, then runs the
# Arm-contributed CortexM7Model in llvm-mca over each hot-kernel region.
#
# Usage: tools/perf_model/codegen_census.sh [out_dir]   (default build/perf_model)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$REPO_ROOT/build/perf_model}"
SRC_ARM="$REPO_ROOT/spectral_engine/synth/backends/arm"
TOOL_DIR="$REPO_ROOT/tools/perf_model"
MCA="${LLVM_MCA:-/opt/homebrew/opt/llvm/bin/llvm-mca}"

CFLAGS=(-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -O3 -ffreestanding
        -isystem "$TOOL_DIR/fs_include"
        -DSPECTRAL_EMBEDDED=1 -DSPECTRAL_ARM_M7=1 -DSPECTRAL_HAS_DUAL_MAC=1
        -I"$REPO_ROOT/spectral_engine" -I"$REPO_ROOT/spectral_engine/core"
        -I"$REPO_ROOT/spectral_engine/synth" -I"$REPO_ROOT/spectral_engine/synth/math"
        -I"$REPO_ROOT/spectral_engine/synth/api" -I"$REPO_ROOT/spectral_engine/runtime"
        -I"$REPO_ROOT/spectral_engine/analysis")

mkdir -p "$OUT"

# --- Layer 0: full-TU census (the production TU, exactly as the daisy target sees it)
arm-none-eabi-gcc "${CFLAGS[@]}" -S "$SRC_ARM/spectral_synth_arm32.c" -o "$OUT/arm32_m7.s"
{
  echo "# Instruction census — spectral_synth_arm32.c @ cortex-m7 -O3 [measured: codegen]"
  echo "# $(arm-none-eabi-gcc --version | head -1)"
  grep -oE '^\s+(smulbb|smlald|smlad|smlabb|qadd16|qsub16|ssat|qadd|qsub|smull|smlal|umull|umlal|mla|mls|vldr|vstr|vmul\.f64|vadd\.f64|pld)' \
    "$OUT/arm32_m7.s" | sed 's/^[[:space:]]*//' | sort | uniq -c | sort -rn
} | tee "$OUT/census.txt"

# --- Layer 2: innermost hot-loop bodies through llvm-mca CortexM7Model.
# Whole inlined kernels are NOT fed to mca (it models straight-line code);
# extract_loops.py reduces each marked kernel to its innermost loop bodies.
arm-none-eabi-gcc "${CFLAGS[@]}" -I"$SRC_ARM" -S "$TOOL_DIR/kernel_wrappers.c" \
  -o "$OUT/kernel_wrappers.s"

python3 "$TOOL_DIR/extract_loops.py" "$OUT/kernel_wrappers.s" "$OUT/loops.mca.s"

"$MCA" -mtriple=thumbv7em-none-eabi -mcpu=cortex-m7 -iterations=100 \
  "$OUT/loops.mca.s" > "$OUT/mca_report.txt"

echo
echo "# Per-loop steady-state [modeled: llvm-mca/CortexM7Model] — full report: $OUT/mca_report.txt"
grep -E "^\[[0-9]+\] Code Region|^Instructions:|^Total Cycles|^IPC|^Block RThroughput" \
  "$OUT/mca_report.txt"
