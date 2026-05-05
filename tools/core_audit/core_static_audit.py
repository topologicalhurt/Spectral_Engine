#!/usr/bin/env python3
"""Static audit checks for high-confidence Spectral Engine core invariants."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
FAILURES: list[str] = []


def read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        FAILURES.append(f"missing {rel}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def require(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def body_of(src: str, fn: str) -> str:
    start = src.find(f"static void {fn}")
    if start < 0:
        start = src.find(f"void {fn}")
    if start < 0:
        return ""
    brace = src.find("{", start)
    if brace < 0:
        return src[start:]
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    return src[start:]


def main() -> int:
    require(not any(ROOT.glob(".spectral_core_audit_backups_*")), "audit backup directories must not be committed")

    gitignore = read(".gitignore")
    require(".spectral_core_audit_backups_*/" in gitignore, "audit backup directory pattern must be ignored")

    osc = read("spectral_engine/core/spectral_osc_formulas.h")
    require("#define SPECTRAL_OSC_FORMULAS_VERSION 3" in osc, "oscillator formula version must be 3")
    require("return sinf(x);" in osc, "spectral_fast_sin_inline must use exact sinf by default")
    require("SPECTRAL_ENABLE_APPROX_TRIG" in osc, "approx trig must be explicitly gated")
    require("Run `make parity-test`" not in osc, "do not reference nonexistent parity-test make target")

    fm = read("spectral_engine/core/spectral_fast_math.c")
    require("return atan2f(y, x);" in fm, "fast_atan2 must be exact by default")
    require("SPECTRAL_ENABLE_APPROX_INV_SQRT" in fm, "approx inverse sqrt must be explicitly gated")

    vo = read("spectral_engine/core/spectral_vector_ops.c")
    require("#if !SPECTRAL_ENABLE_APPROX_ATAN2" in vo, "vector phase extraction must gate approximate atan2")
    require("phase[i] = atan2f(im, re);" in vo, "magsq+phase extraction must use exact atan2f by default")

    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")
    require("const size_t pair_count = n_frames - 1u" in fused, "fused path must iterate explicit adjacent frame pairs")
    require("pair_start" in fused and "pair_end" in fused, "fused path must process explicit pair ranges")
    require("fctx.row = row_curr" in fused and "fctx.next_row = row_next" in fused and "fctx.phase_row = phase_curr" in fused,
            "fused path must track current row against next row with current phase row")
    require("float t_hop = (float)pair * (float)hop" in fused, "fused t_hop must be based on the actual pair index")
    require("t - 1" not in fused, "fused path must not use stale t-1 indexing")
    require("row_prev" not in fused and "phase_prev" not in fused, "fused path should not keep ambiguous prev/current row names")
    region_start = fused.find("Global maximum discovery")
    region_end = fused.find("tracker = spectral_tracker_create")
    if region_start >= 0 and region_end >= 0:
        require("SPECTRAL_CUSTOM_FAST_MATH_MODE" not in fused[region_start:region_end],
                "fused max discovery must not be disabled by fast-math mode")
    else:
        require(False, "fused analysis max-discovery region not found")

    synth = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    for fn in ["segment_fn_wavetable_float", "segment_fn_native_timbre", "segment_fn_native_wavetable"]:
        body = body_of(synth, fn)
        require(body, f"missing {fn}")
        require("fade_envelope" in body, f"{fn} must apply segment fade envelope")

    wavetable = read("spectral_engine/core/spectral_wavetable.c")
    require('#include "spectral_osc_formulas.h"' in wavetable, "wavetable builtins must include canonical oscillator formulas")
    require("2.0 * t - 1.0" not in wavetable, "wavetable saw must not use independent phase convention")
    require("if (!bank) return SPECTRAL_SAMPLE_ZERO;" in wavetable, "wavetable timbre lookup must guard NULL bank")

    metal = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    require("SPECTRAL_OSC_FORMULAS_VERSION == 3" in metal, "Metal oscillator formula guard must be version 3")
    require("SPECTRAL_METAL_FAST_MATH" in metal, "Metal fast math must be explicitly gated")

    metal_src_owner = read("spectral_engine/core/oscillator.c")
    require("float norm = p * INV_TWO_PI" not in metal_src_owner, "Metal normalize string must not contain unused norm variable")
    require("return p - TWO_PI * floor(p * INV_TWO_PI + 0.5f);" in metal_src_owner,
            "Metal normalize string must preserve zero phase")

    header = read("spectral_engine/analysis/spectral_peak_track.h")
    require("VM overcommit" not in header, "peak tracker public comment must not advertise VM overcommit")


    synth_internal = read("spectral_engine/core/spectral_synth_internal.c")
    require("SpectralError synth_validate_params" in synth_internal, "synthesis must validate public stretch/pitch params centrally")
    require("SPECTRAL_MAX_STRETCH" in synth_internal and "SPECTRAL_MIN_PITCH" in synth_internal,
            "synthesis param validation must use canonical configured bounds")
    require("!spectral_is_finite_f32(s->omega)" in synth_internal and "!spectral_is_finite_f32(s->amp)" in synth_internal,
            "segment loop params must reject non-finite segment fields")
    require("stretch > SPECTRAL_MAX_STRETCH" in synth_internal[synth_internal.find("gpu_tile_preprocess"):],
            "GPU tile preprocessing must reject invalid stretch before tile math")
    require("spectral_gpu_segment_tile_span" in synth_internal,
            "GPU tile preprocessing must use a single canonical segment-to-tile helper")
    require("floor(start / tile_size_d)" in synth_internal and "ceil(end / tile_size_d)" in synth_internal,
            "GPU tile span helper must use canonical half-open interval tile bounds")
    require("int start_tile = (int)(start / tile_size)" not in synth_internal and
            "int end_tile = (int)(end / tile_size)" not in synth_internal,
            "GPU tile preprocessing must not cast unbounded float tile indices to int")
    require("if (total_refs > 0u)" in synth_internal,
            "GPU tile preprocessing must allow zero active references without malloc(0) failure")
    require("tile_size == (uint32_t)SPECTRAL_GPU_TILE_SIZE" in synth_internal,
            "GPU tile cache reuse must be gated to the canonical cache tile size")

    synth_header = read("spectral_engine/core/spectral_synth_internal.h")
    require("SpectralError error" in synth_header and "synth_preflight_native" in synth_header,
            "SynthPreflight must carry error status and native preflight")

    cuda = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")
    require("if (!pf.ok) return pf.error;" in cuda, "CUDA preflight must propagate parameter errors")

    metal_src = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    require("if (!pf.ok) return pf.error;" in metal_src, "Metal preflight must propagate parameter errors")

    analysis = read("spectral_engine/analysis/spectral_analysis.c")
    require("SPECTRAL_MIN_SAMPLE_RATE" in analysis and "SPECTRAL_MIN_FFT_SIZE" in analysis,
            "analysis input validation must use canonical sample-rate and FFT-size bounds")

    fft = read("spectral_engine/analysis/spectral_analysis_fft.c")
    require("n_freqs != (n_fft / 2u + 1u)" in fft and "SPECTRAL_MIN_FFT_SIZE" in fft, "FFT resource allocation must validate size, power-of-two shape, and frequency-bin shape")

    fft_owner = read("spectral_engine/analysis/spectral_analysis_fft.c")
    require("res->fft_setups = spectral_calloc_array" in fft_owner and
            "res->fft_plans = spectral_calloc_array" in fft_owner,
            "FFT resource pointer arrays must be zero-initialized for partial-failure cleanup")
    require("fail:\n    spectral_fft_resources_free(res);\n    return 0;" in fft_owner,
            "FFT resource allocation must use a local cleanup path")
    require("if (!res) return;" in fft_owner and "memset(res, 0, sizeof(*res));" in fft_owner,
            "FFT resource free must be null-safe and zero released state")



    fft_src = read("spectral_engine/analysis/spectral_analysis_fft.c")
    require("#include <limits.h>" in fft_src, "FFT resource allocator must include limits.h for INT_MAX guard")
    require("if (!res) return 0;" in fft_src, "FFT allocator must handle NULL before zeroing")
    require("memset(res, 0, sizeof(*res));\n\n    if (n_threads < 1" in fft_src,
            "FFT allocator must zero resource state before any shape-failure return")
    require("n_fft > (size_t)INT_MAX" in fft_src,
            "FFT allocator must reject FFT sizes that cannot be passed to FFTW int APIs")

    full_src = read("spectral_engine/analysis/spectral_analysis_full.c")
    require("SpectralFftResources res = {0};" in full_src,
            "full analysis path must initialize FFT resources before allocation attempts")


    windows_h = read("spectral_engine/core/spectral_windows.h")
    windows_c = read("spectral_engine/core/spectral_windows.c")
    require("typedef void (*SpectralWindowGenerateFn)" in windows_h and
            "typedef float (*SpectralWindowInterpMagsqFn)" in windows_h,
            "window API must expose descriptor callback types")
    require("SpectralWindowDescriptor" in windows_h and
            "SpectralWindowMetrics" in windows_h,
            "window API must expose descriptor registry and metrics structs")
    require("SPECTRAL_WINDOW_METRIC_POSITIVE_BIN_SCALE_VALID" in windows_h and
            "SPECTRAL_WINDOW_METRIC_ENDPOINT_BIN_SCALE_VALID" in windows_h and
            "SPECTRAL_WINDOW_METRIC_ENBW_VALID" in windows_h,
            "window metrics must expose validity flags for conditional metrics")
    require("spectral_window_descriptor(SpectralWindowType type)" in windows_h and
            "spectral_window_descriptor_at(size_t index)" in windows_h and
            "spectral_window_descriptor_count(void)" in windows_h and
            "spectral_window_find_by_id" in windows_h,
            "window API must expose static descriptor registry lookup helpers")
    require("spectral_window_metrics" in windows_h,
            "window API must expose universal sample-derived metrics")
    require("static const SpectralWindowDescriptor spectral_window_descriptors[]" in windows_c and
            '"hann"' in windows_c and '"hamming"' in windows_c and
            '"blackman"' in windows_c and '"rectangular"' in windows_c,
            "window implementation must register built-ins through descriptor table")
    require("desc->generate(window, length)" in windows_c,
            "spectral_window_generate must delegate through the descriptor registry")
    require("spectral_window_positive_bin_magsq_scale" in windows_h,
            "window API must expose positive-bin magnitude-squared calibration")
    require("spectral_window_endpoint_bin_magsq_scale" in windows_h and
            "endpoint_bin_magsq_scale" in windows_h,
            "window API must expose endpoint magnitude-squared calibration")
    require("vDSP_HANN_NORM" not in windows_c,
            "vDSP Hann path must not request normalized window when API promises conventional windows")
    require("vDSP_HANN_DENORM" in windows_c,
            "vDSP Hann path must use conventional unnormalized Hann")
    require("positive_amp_scale_d = 2.0 / (double)metrics.sum" in windows_c,
            "positive-bin amplitude calibration must derive from the real-sinusoid 2/sum(window) scale")
    require("endpoint_amp_scale_d = 1.0 / (double)metrics.sum" in windows_c,
            "endpoint amplitude calibration must derive from the real-sinusoid 1/sum(window) scale")
    require("isfinite(positive_amp_scale)" in windows_c and
            "isfinite(positive_magsq_scale)" in windows_c and
            "isfinite(endpoint_amp_scale)" in windows_c and
            "isfinite(endpoint_magsq_scale)" in windows_c,
            "window scale validity flags must be gated on finite derived scales, not only finite window sum")
    require("metrics.flags |= SPECTRAL_WINDOW_METRIC_POSITIVE_BIN_SCALE_VALID" in windows_c and
            "SPECTRAL_WINDOW_METRIC_ENDPOINT_BIN_SCALE_VALID" in windows_c and
            "metrics.flags |= SPECTRAL_WINDOW_METRIC_ENBW_VALID" in windows_c,
            "window metrics must set validity flags when derived metrics are usable")

    internal_h = read("spectral_engine/analysis/spectral_analysis_internal.h")
    require("float endpoint_bin_magsq_scale;" in internal_h and
            "float positive_bin_magsq_scale;" in internal_h,
            "FFT resources must carry endpoint and positive-bin scale state")
    require("spectral_fft_resources_set_magsq_scales" in internal_h,
            "FFT resources must expose explicit endpoint-aware magnitude-squared scale setter")

    fft_scale_src = read("spectral_engine/analysis/spectral_analysis_fft.c")
    require("res->endpoint_bin_magsq_scale = 1.0f;" in fft_scale_src and
            "res->positive_bin_magsq_scale = 1.0f;" in fft_scale_src,
            "FFT resources must default endpoint and positive-bin scales to raw magnitude-squared values")
    require("spectral_fft_apply_magsq_scales" in fft_scale_src and
            "spectral_fft_trackable_magsq_max" in fft_scale_src,
            "FFT frame extraction must apply endpoint-aware scales and compute trackable-bin maxima")
    require("magsq[0] *= endpoint_scale" in fft_scale_src and
            "magsq[n_freqs - 1u] *= endpoint_scale" in fft_scale_src and
            "magsq[i] *= positive_scale" in fft_scale_src,
            "FFT scaling must treat DC/Nyquist endpoints separately from interior positive bins")
    require("*frame_max = spectral_fft_trackable_magsq_max(magsq, n_freqs)" in fft_scale_src and
            "*frame_max *= scale" not in fft_scale_src,
            "FFT frame maximum must be recomputed from scaled trackable bins, not uniformly scaled")
    require("void spectral_fft_resources_set_magsq_scales" in
            fft_scale_src[:fft_scale_src.find("int spectral_fft_resources_alloc")],
            "FFT magnitude scale setter must not be nested inside resource allocation")
    require("static void spectral_fft_apply_magsq_scales" in
            fft_scale_src[:fft_scale_src.find("int spectral_fft_resources_alloc")],
            "FFT magnitude scale helper must not be nested inside resource allocation")

    full_src = read("spectral_engine/analysis/spectral_analysis_full.c")
    fused_src = read("spectral_engine/analysis/spectral_analysis_fused.c")
    require("spectral_window_hann(window_func" not in full_src and
            "spectral_window_hann(window_func" not in fused_src,
            "analysis paths must not hard-code Hann generation directly")
    require("spectral_window_generate(window_func, (size_t)n_fft, SPECTRAL_WINDOW_HANN)" in full_src and
            "spectral_window_generate(window_func, (size_t)n_fft, SPECTRAL_WINDOW_HANN)" in fused_src,
            "analysis paths must route default Hann generation through registry wrapper")
    require("window_metrics = spectral_window_metrics(window_func, (size_t)n_fft)" in full_src and
            "window_metrics.endpoint_bin_magsq_scale" in full_src and
            "window_metrics.positive_bin_magsq_scale" in full_src,
            "full-matrix analysis must calibrate endpoint and positive-bin magnitudes from window metrics")
    require("window_metrics = spectral_window_metrics(window_func, (size_t)n_fft)" in fused_src and
            "window_metrics.endpoint_bin_magsq_scale" in fused_src and
            "window_metrics.positive_bin_magsq_scale" in fused_src,
            "fused analysis must calibrate endpoint and positive-bin magnitudes from window metrics")

    ai_canon = read("docs/core_audit/AI_CANON.md")
    require("Named techniques and paper-backed claims need sources" in ai_canon and
            "source link" in ai_canon and "technical explanation" in ai_canon,
            "AI canon must require links or technical explanations for named techniques and paper-backed claims")
    require("Reuse existing core math utilities before adding formulas" in ai_canon and
            "fast_sqrt" in ai_canon and
            "comparison test between approximate and canonical outputs" in ai_canon,
            "AI canon must require utility reuse and approximation/reference comparison tests")


    pass8_notes = read("docs/core_audit/PATCH_NOTES_PASS8.md")
    require("FFT resources now carry an explicit `magsq_scale`" not in pass8_notes,
            "pass 8 notes must not describe stale single-scale FFT resource contract")
    require("applies the scale to every magnitude-squared row" not in pass8_notes,
            "pass 8 notes must not describe stale uniform all-bin scaling")
    require("endpoint/interior-bin" in pass8_notes and "trackable interior bins" in pass8_notes,
            "pass 8 notes must document endpoint-aware scaling and trackable-bin maxima")
    require("raw_endpoint_magnitude = sinusoid_peak_amplitude * sum(window)" in pass8_notes and
            "DC and Nyquist" in pass8_notes,
            "pass 8 notes must document endpoint-bin calibration")
    require("raw_positive_bin_magnitude = sinusoid_peak_amplitude * sum(window) / 2" in pass8_notes,
            "pass 8 notes must document interior positive-bin calibration")
    require("endpoint_amp_scale_d = 1.0 / (double)metrics.sum" in windows_c and
            "endpoint_magsq_scale_d = endpoint_amp_scale_d * endpoint_amp_scale_d" in windows_c,
            "pass 8 implementation must preserve endpoint magnitude-squared scale")
    require("positive_amp_scale_d = 2.0 / (double)metrics.sum" in windows_c and
            "positive_magsq_scale_d = positive_amp_scale_d * positive_amp_scale_d" in windows_c,
            "pass 8 implementation must preserve interior positive-bin magnitude-squared scale")


    track_h = read("spectral_engine/analysis/spectral_peak_track.h")
    track_internal_h = read("spectral_engine/analysis/spectral_peak_track_internal.h")
    track_c = read("spectral_engine/analysis/spectral_peak_track.c")
    interp_c = read("spectral_engine/analysis/spectral_peak_interp.c")
    fused_src = read("spectral_engine/analysis/spectral_analysis_fused.c")
    pass10_notes = read("docs/core_audit/PATCH_NOTES_PASS10.md")

    require("spectral_tracker_set_window_descriptor" in track_h,
            "peak tracker public API must allow analysis code to bind the active window descriptor")
    require("spectral_track_peaks_with_window_descriptor" in track_h,
            "raw single-shot tracking API must expose explicit window descriptor binding")
    require("SpectralWindowInterpMagsqFn interp_magsq;" in track_internal_h,
            "peak tracker internals must store the active window interpolation callback")
    require("tracker->interp_magsq = spectral_window_interp_magsq_parabolic;" in track_c,
            "peak tracker must initialize a safe default interpolation callback")
    require("spectral_tracker_set_window_descriptor(tracker, window_desc);" in track_c,
            "explicit single-shot tracking API must bind the caller-provided window descriptor")
    require("spectral_track_peaks_with_window_descriptor" in full_src and
            "spectral_window_descriptor(SPECTRAL_WINDOW_HANN)" in full_src,
            "full-matrix analysis must call explicit single-shot tracking with the default Hann descriptor")
    require("spectral_tracker_set_window_descriptor(tracker, spectral_window_descriptor(SPECTRAL_WINDOW_HANN));" in fused_src,
            "fused tracker creation must bind the default Hann descriptor")
    require("left = row[cf - 1u];" in interp_c and
            "right = row[cf + 1u];" in interp_c and
            "!isfinite(left) || !isfinite(curr) || !isfinite(right)" in interp_c,
            "candidate validation must reject non-finite current-frame interpolation neighborhoods")
    require("spectral_window_interp_magsq_parabolic(left, curr, right)" not in interp_c,
            "peak emission must not hard-code the global parabolic interpolation helper")
    require("!isfinite(threshsq) || threshsq < 0.0f" in interp_c,
            "candidate validation must reject invalid power thresholds")
    require("0.5 * (log(left) - log(right))" in pass10_notes and
            "https://doi.org/10.1109/PROC.1978.10837" in pass10_notes and
            "https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html" in pass10_notes and
            "https://doi.org/10.1109/78.295186" in pass10_notes and
            "https://doi.org/10.1109/MSP.2007.361611" in pass10_notes and
            "https://doi.org/10.1109/LSP.2011.2136378" in pass10_notes,
            "pass 10 notes must document and cite the estimator contract")


    estimator_h = read("spectral_engine/analysis/spectral_peak_estimator.h")
    estimator_c = read("spectral_engine/analysis/spectral_peak_estimator.c")
    config_h = read("spectral_engine/core/spectral_config.h")
    windows_c = read("spectral_engine/core/spectral_windows.c")
    fast_math_h = read("spectral_engine/core/spectral_fast_math.h")
    fast_math_c = read("spectral_engine/core/spectral_fast_math.c")
    manifest = read("spectral_engine/cmake/source-manifest.cmake")
    track_h = read("spectral_engine/analysis/spectral_peak_track.h")
    track_internal_h = read("spectral_engine/analysis/spectral_peak_track_internal.h")
    track_c = read("spectral_engine/analysis/spectral_peak_track.c")
    interp_c = read("spectral_engine/analysis/spectral_peak_interp.c")
    pass11_notes = read("docs/core_audit/PATCH_NOTES_PASS11.md")
    estimator_contract_test = read("tests/core_math/test_core_pass11_peak_estimator_contract.py")
    estimator_bench = read("tools/core_audit/peak_estimator_bench.c")

    require("spectral_peak_estimator.c" in manifest,
            "peak estimator module must be compiled through source manifest")
    require("SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC" in estimator_h and
            "SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX" in estimator_h and
            "SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX" in estimator_h and
            "SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND" in estimator_h,
            "peak estimator API must expose baseline and advanced estimator candidates")
    require("spectral_peak_estimate_validated" in estimator_h and
            "spectral_peak_candan_correction_for_n_freqs" in estimator_h and
            "float candan_correction;" in estimator_h,
            "peak estimator API must expose tracker-only validated path and Candan precompute")
    require("SPECTRAL_ENABLE_APPROX_PEAK_LOG" in config_h,
            "peak log approximation must be gated behind an explicit opt-in flag")
    require("spectral_peak_complex_offset_jacobsen" in estimator_c and
            "spectral_peak_offset_candan" in estimator_c and
            "spectral_peak_offset_quinn_second" in estimator_c,
            "peak estimator implementation must include Jacobsen, Candan and Quinn candidates")
    require("spectral_peak_reconstruct_triplet" in estimator_c and
            "spectral_peak_sincosf" in estimator_c and
            "SPECTRAL_ENABLE_APPROX_TRIG" in estimator_c and
            "fast_sqrt(" in estimator_c and
            "fast_peak_log(" in estimator_c and
            "0x5f3759df" not in estimator_c,
            "peak estimator performance paths must share triplet reconstruction and reuse central fast math")
    require("#include <limits.h>" in estimator_c and
            "spectral_peak_best_next_valid" in estimator_c and
            "input->best_next_bin < 0" in estimator_c and
            "best_next + 1u < input->bin" in estimator_c,
            "peak estimator must validate best_next_bin before df arithmetic")
    require("FAST_MATH_HOT float fast_peak_log(float x);" in fast_math_h and
            "float fast_peak_log(float x)" in fast_math_c and
            "SPECTRAL_ENABLE_APPROX_PEAK_LOG" in fast_math_c and
            "https://standards.ieee.org/standard/754-2019.html" in fast_math_c and
            "https://dlmf.nist.gov/4.37.E25" in fast_math_c and
            "https://dlmf.nist.gov/4.37.E31" in fast_math_c and
            "0x5f3759df" in fast_math_c,
            "approximate peak log and inverse sqrt implementations must live in central fast_math utilities with source citations")
    require("log_lc = fast_peak_log(left / center)" in windows_c and
            "log_rc = fast_peak_log(right / center)" in windows_c and
            "spectral_window_peak_logf" not in windows_c,
            "log-power parabolic interpolation must use the two-log ratio form")
    require("https://doi.org/10.1109/PROC.1978.10837" in estimator_c and
            "https://doi.org/10.1109/78.295186" in estimator_c and
            "https://doi.org/10.1109/MSP.2007.361611" in estimator_c and
            "https://doi.org/10.1109/LSP.2011.2136378" in estimator_c,
            "peak estimator implementation must cite estimator source basis")
    require("return SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;" in estimator_c,
            "AUTO estimator must preserve the conservative log-parabolic default")
    require("raw_offset = interp(left, center, right);" in estimator_c and
            "if (!isfinite(raw_offset)) return 0;" in estimator_c,
            "custom interpolation callbacks must fail on non-finite output")
    require("SPECTRAL_PEAK_ESTIMATE_USED_FALLBACK" in estimator_c,
            "advanced estimators must have explicit fallback reporting")
    require("spectral_tracker_set_peak_estimator" in track_h and
            "SpectralPeakEstimatorType peak_estimator;" in track_internal_h,
            "tracker must expose and store the active peak estimator policy")
    require("float peak_candan_correction;" in track_internal_h and
            "tracker->peak_candan_correction = spectral_peak_candan_correction_for_n_freqs(n_freqs);" in track_c and
            "estimate_input.candan_correction = tracker->peak_candan_correction;" in interp_c,
            "tracker must precompute and pass Candan correction instead of recomputing it per peak")
    require("tracker->peak_estimator = SPECTRAL_PEAK_ESTIMATOR_DEFAULT;" in track_c,
            "tracker must initialize estimator policy from canonical default")
    require("SpectralPeakEstimateInput estimate_input" in interp_c and
            "spectral_peak_estimate_validated(&estimate_input, &estimate)" in interp_c,
            "segment emission must delegate to the tracker-validated estimator hot path")
    require("spectral_window_interp_magsq_parabolic(left, curr, right)" not in interp_c,
            "segment emission must not hard-code the old parabolic helper")
    require("0.02` bins" in pass11_notes and
            "https://doi.org/10.1109/MSP.2007.361611" in pass11_notes and
            "spectral_track_peaks_with_window_descriptor" in pass11_notes and
            "SPECTRAL_ENABLE_APPROX_PEAK_LOG" in pass11_notes and
            "two-log" in pass11_notes,
            "pass 11 notes must document estimator tolerance, citations, explicit raw tracking API and performance flags")
    require("HANN_LOG_PARABOLIC_MAX_ERROR_BINS = 0.02" in estimator_contract_test and
            "spectral_peak_estimate(&input, &out)" in estimator_contract_test and
            "nan_offset" in estimator_contract_test and
            "INT_MIN" in estimator_contract_test and
            "test_raw_tracker_descriptor_changes_single_shot_omega" in estimator_contract_test and
            "spectral_track_peaks_with_window_descriptor" in estimator_contract_test and
            "test_two_log_ratio_matches_three_log_formula" in estimator_contract_test and
            "test_approximation_flags_stay_within_peak_error_contract" in estimator_contract_test and
            "test_peak_estimator_benchmark_harness_reports_all_methods" in estimator_contract_test,
            "peak estimator contract tests must cover Hann bias, callback binding, raw tracker descriptor binding, malformed neighborhoods, exact reduction, approximation flags and bench harness")
    require("BenchExpected" in estimator_bench and
            "Correctness accounting is intentionally outside the timed loop" in estimator_bench and
            "spectral_peak_estimate_validated(&inputs[case_idx], &out)" in estimator_bench and
            "p50_ns=" in estimator_bench and
            "p95_ns=" in estimator_bench and
            "max_offset_err=" in estimator_bench and
            "fallback_rate=" in estimator_bench,
            "peak estimator benchmark must time estimator path separately from emitted-field error and fallback reporting")


    estimator_c_pass12 = read("spectral_engine/analysis/spectral_peak_estimator.c")
    require("Keep these scalar-contract checks unconditional" in estimator_c_pass12,
            "validated peak estimator path must keep scalar-context finite checks")
    require("custom descriptor callback" in estimator_c_pass12,
            "descriptor interpolation callbacks must not receive NaN/negative magnitudes")
    require("if (!spectral_peak_finite_nonnegative(magsq) || !isfinite(phase))" in estimator_c_pass12,
            "complex coefficient reconstruction must always validate magnitude and phase")

    if FAILURES:
        for f in FAILURES:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("core static audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
