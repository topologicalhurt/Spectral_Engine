#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  benchmark_spectral.sh --binary <path> --input <wav> [--runs N] [--mode normal|cache] [--cache-dir <dir>] -- [legacy positional args]

Examples:
  benchmark_spectral.sh --binary ../bin/spectral_arm64_metal_desktop --input ../resources/motormouth_recites_shakespeare_he_saw_the_cat.wav --runs 6 --mode normal -- 0 1.0 0 4096 128 -90 8 1
  benchmark_spectral.sh --binary ../bin/spectral_arm64_metal_desktop --input ../resources/motormouth_recites_shakespeare_he_saw_the_cat.wav --runs 6 --mode cache  -- 0 1.0 0 4096 128 -90 8 1
EOF
}

median() {
    if [ "$#" -eq 0 ]; then
        echo "nan"
        return
    fi
    printf "%s\n" "$@" | LC_ALL=C sort -n | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) { print "nan"; exit }
            if ((NR % 2) == 1) printf "%.3f\n", a[(NR + 1) / 2];
            else printf "%.3f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2.0;
        }'
}

mean() {
    if [ "$#" -eq 0 ]; then
        echo "nan"
        return
    fi
    printf "%s\n" "$@" | awk '{ s += $1 } END { if (NR > 0) printf "%.3f\n", s / NR; else print "nan"; }'
}

parse_normal_timing() {
    local log_file="$1"
    awk '
        /^FFT: .* Track: .* Synth: .* Norm: .* Total:/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "FFT:")   { v = $(i + 1); gsub(/ms/, "", v); fft = v; }
                if ($i == "Track:") { v = $(i + 1); gsub(/ms/, "", v); track = v; }
                if ($i == "Synth:") { v = $(i + 1); gsub(/ms/, "", v); synth = v; }
                if ($i == "Norm:")  { v = $(i + 1); gsub(/ms/, "", v); norm = v; }
                if ($i == "Total:") { v = $(i + 1); gsub(/ms/, "", v); total = v; }
            }
        }
        END {
            if (total != "") printf "%.6f %.6f %.6f %.6f %.6f\n", fft, track, synth, norm, total;
        }' "$log_file"
}

parse_cache_timing() {
    local log_file="$1"
    awk '
        /^Segment-binary synth run:/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "Synth") { v = $(i + 1); gsub(/ms/, "", v); synth = v; }
                if ($i == "Norm")  { v = $(i + 1); gsub(/ms/, "", v); norm = v; }
                if ($i == "Total") { v = $(i + 1); gsub(/ms/, "", v); seg_total = v; }
            }
        }
        /^Cache-mode end-to-end total:/ {
            v = $NF;
            gsub(/ms/, "", v);
            end_to_end = v;
        }
        END {
            if (end_to_end != "") printf "%.6f %.6f %.6f %.6f\n", synth, norm, seg_total, end_to_end;
        }' "$log_file"
}

binary=""
input=""
runs=6
mode="normal"
cache_dir=""
args=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)
            binary="$2"
            shift 2
            ;;
        --input)
            input="$2"
            shift 2
            ;;
        --runs)
            runs="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --cache-dir)
            cache_dir="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            args=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$binary" ] || [ -z "$input" ]; then
    echo "Error: --binary and --input are required." >&2
    usage >&2
    exit 2
fi

if [ ! -x "$binary" ]; then
    echo "Error: binary not executable: $binary" >&2
    exit 1
fi

if [ ! -f "$input" ]; then
    echo "Error: input file not found: $input" >&2
    exit 1
fi

if ! [[ "$runs" =~ ^[0-9]+$ ]] || [ "$runs" -lt 1 ]; then
    echo "Error: --runs must be a positive integer." >&2
    exit 2
fi

if [ "$mode" != "normal" ] && [ "$mode" != "cache" ]; then
    echo "Error: --mode must be 'normal' or 'cache'." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
if [ -z "$cache_dir" ]; then
    cache_dir="$repo_root/output/cache"
fi

if [ "$mode" = "cache" ]; then
    mkdir -p "$cache_dir"
    input_stem="$(basename "$input")"
    input_stem="${input_stem%.*}"
    n_fft="${args[3]:-4096}"
    hop="${args[4]:-128}"
    db_thresh="${args[5]:--90}"
    start_sec="${args[8]:-0}"
    end_sec="${args[9]:--1}"
    db10="$(awk -v d="$db_thresh" 'BEGIN { printf "%d", d * 10.0 }')"
    start_ms="$(awk -v s="$start_sec" 'BEGIN { printf "%d", s * 1000.0 }')"
    end_ms="$(awk -v e="$end_sec" 'BEGIN { printf "%d", e * 1000.0 }')"
    cache_file="$cache_dir/${input_stem}_n${n_fft}_h${hop}_db${db10}_s${start_ms}_e${end_ms}.segbin"
    rm -f "$cache_file"
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/spectral-bench.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

totals=()
ffts=()
tracks=()
synths=()
norms=()
segbin_totals=()
first_total=""

echo "Benchmark mode: $mode"
echo "Binary: $binary"
echo "Input:  $input"
echo "Runs:   $runs"
echo ""

for ((i = 1; i <= runs; i++)); do
    log="$tmpdir/run_${i}.log"

    if [ "$mode" = "cache" ]; then
        if ! "$binary" "$input" "${args[@]}" --cache >"$log" 2>&1; then
            echo "Run $i failed. Last log lines:" >&2
            tail -n 60 "$log" >&2 || true
            exit 1
        fi

        read -r synth_ms norm_ms seg_total_ms total_ms < <(parse_cache_timing "$log")
        if [ -z "${total_ms:-}" ]; then
            echo "Run $i: failed to parse cache timing output." >&2
            tail -n 80 "$log" >&2 || true
            exit 1
        fi

        totals+=("$total_ms")
        synths+=("$synth_ms")
        norms+=("$norm_ms")
        segbin_totals+=("$seg_total_ms")
        printf "run %02d  total=%8.2fms  segbin=%8.2fms  synth=%8.2fms  norm=%8.2fms\n" \
            "$i" "$total_ms" "$seg_total_ms" "$synth_ms" "$norm_ms"
    else
        if ! "$binary" "$input" "${args[@]}" >"$log" 2>&1; then
            echo "Run $i failed. Last log lines:" >&2
            tail -n 60 "$log" >&2 || true
            exit 1
        fi

        read -r fft_ms track_ms synth_ms norm_ms total_ms < <(parse_normal_timing "$log")
        if [ -z "${total_ms:-}" ]; then
            echo "Run $i: failed to parse normal timing output." >&2
            tail -n 80 "$log" >&2 || true
            exit 1
        fi

        totals+=("$total_ms")
        ffts+=("$fft_ms")
        tracks+=("$track_ms")
        synths+=("$synth_ms")
        norms+=("$norm_ms")
        printf "run %02d  total=%8.2fms  fft=%8.2fms  track=%8.2fms  synth=%8.2fms  norm=%8.2fms\n" \
            "$i" "$total_ms" "$fft_ms" "$track_ms" "$synth_ms" "$norm_ms"
    fi

    if [ "$i" -eq 1 ]; then
        first_total="$total_ms"
    fi
done

all_median="$(median "${totals[@]}")"
all_mean="$(mean "${totals[@]}")"

warm_median="nan"
warm_mean="nan"
if [ "$runs" -gt 1 ]; then
    warm_totals=("${totals[@]:1}")
    warm_median="$(median "${warm_totals[@]}")"
    warm_mean="$(mean "${warm_totals[@]}")"
fi

echo ""
echo "--- Summary ---"
printf "Total ms: first=%.3f median=%.3f mean=%.3f warm_median=%s warm_mean=%s\n" \
    "$first_total" "$all_median" "$all_mean" "$warm_median" "$warm_mean"

if [ "$mode" = "cache" ]; then
    seg_median="$(median "${segbin_totals[@]}")"
    synth_median="$(median "${synths[@]}")"
    norm_median="$(median "${norms[@]}")"
    printf "Segment-binary ms: total_median=%s synth_median=%s norm_median=%s\n" \
        "$seg_median" "$synth_median" "$norm_median"
else
    fft_median="$(median "${ffts[@]}")"
    track_median="$(median "${tracks[@]}")"
    synth_median="$(median "${synths[@]}")"
    norm_median="$(median "${norms[@]}")"
    printf "Stage medians ms: fft=%s track=%s synth=%s norm=%s\n" \
        "$fft_median" "$track_median" "$synth_median" "$norm_median"
fi
