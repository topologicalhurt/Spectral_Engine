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

    for pass_num in range(1, 53):
        require((ROOT / f"docs/core_audit/PATCH_NOTES_PASS{pass_num}.md").exists(),
                f"core audit pass {pass_num} notes must use numeric PATCH_NOTES_PASS{pass_num}.md naming")
    require(not (ROOT / "docs/core_audit/PATCH_NOTES.md").exists(),
            "base PATCH_NOTES.md must be normalized to PATCH_NOTES_PASS1.md")
    require(not (ROOT / "docs/core_audit/PATCH_NOTES_INTERIM_PASS3.md").exists(),
            "interim pass 3 notes must be normalized to PATCH_NOTES_PASS3.md")
    require(not (ROOT / "docs/core_audit/PATCH_NOTES_PASS17B.md").exists(),
            "letter-suffixed pass 17 notes must be normalized to PATCH_NOTES_PASS17.md")
    require(not (ROOT / "docs/core_audit/PASS13_ESTIMATOR_VALIDATION_MATRIX.md").exists(),
            "pass 13 validation matrix must live inside PATCH_NOTES_PASS13.md")

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
    require("Patch notes are the canonical pass record" in ai_canon and
            "PATCH_NOTES_PASS<N>.md" in ai_canon and
            "deleting a sidecar file cannot break core tests" in ai_canon,
            "AI canon must require numeric pass-note names and inline pass artifacts")

    academic_sources = read("docs/core_audit/ACADEMIC_SOURCES.md")
    require("https://doi.org/10.1109/TIT.1974.1055282" in academic_sources and
            "https://standards.ieee.org/standard/754-2019.html" in academic_sources and
            "https://dlmf.nist.gov/4.37.E25" in academic_sources,
            "academic sources must cite estimator and fast-log source basis")

    discipline_findings = read("docs/core_audit/DISCIPLINE_FINDINGS.md")
    require("best_next_bin" in discipline_findings and
            "Benchmark timing must exclude reference/error bookkeeping" in discipline_findings and
            "Patch notes are the durable review record" in discipline_findings,
            "discipline findings must cover estimator safety, benchmark timing and audit-doc hygiene")


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
    require("spectral_peak_model_for_window(" in track_c and
            "peak_model.estimator = estimator;" in track_c and
            "spectral_tracker_set_peak_model(tracker, &peak_model)" in track_c,
            "explicit single-shot tracking API must bind the caller-provided window descriptor through a resolved model")
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


    pass13_sweep = read("tools/core_audit/peak_estimator_sweep.c")
    require("SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND" in pass13_sweep and
            "max_abs_err" in pass13_sweep and
            "fallback_rate" in pass13_sweep,
            "pass 13 estimator sweep must report all estimator candidates and error/fallback metrics")
    require("sweep_dft_triplet" in pass13_sweep and
            "spectral_window_generate" in pass13_sweep,
            "pass 13 estimator sweep must generate windowed DFT triplets from synthetic tones")
    pass13_test = read("tests/core_math/test_core_pass13_peak_estimator_sweep.py")
    require("hann_log[\"max\"] <= 0.035" in pass13_test,
            "pass 13 test must protect Hann/log-parabolic high-SNR ground-truth accuracy")
    pass13_notes = read("docs/core_audit/PATCH_NOTES_PASS13.md")
    require("Validation matrix" in pass13_notes and
            "Offset sweep" in pass13_notes and
            "Hann + log-parabolic + 120 dB SNR" in pass13_notes and
            "PASS13_ESTIMATOR_VALIDATION_MATRIX" not in pass13_notes,
            "pass 13 notes must inline the estimator validation matrix")


    estimator_h_pass14 = read("spectral_engine/analysis/spectral_peak_estimator.h")
    estimator_c_pass14 = read("spectral_engine/analysis/spectral_peak_estimator.c")
    interp_h_pass14 = read("spectral_engine/analysis/spectral_peak_interp.h")
    interp_c_pass14 = read("spectral_engine/analysis/spectral_peak_interp.c")
    track_c_pass14 = read("spectral_engine/analysis/spectral_peak_track.c")
    require("const float* next_magsq_row;" in estimator_h_pass14 and
            "float next_bin_offset;" in estimator_h_pass14,
            "pass 14 estimator API must expose next-frame row and next sub-bin offset")
    require("SPECTRAL_PEAK_ESTIMATE_TEMPORAL_DF_REFINED" in estimator_h_pass14,
            "pass 14 estimator flags must report refined temporal df")
    require("spectral_peak_estimate_next_offset_magsq" in estimator_c_pass14 and
            "bin_delta = ((float)input->best_next_bin + next_offset)" in estimator_c_pass14,
            "pass 14 estimator must refine df from next-frame sub-bin offset")
    require("const float* __restrict__ next_row" in interp_h_pass14 and
            "estimate_input.next_magsq_row = next_row;" in interp_c_pass14,
            "pass 14 tracker emission must pass next-row magnitudes into estimator")
    require("spectral_tracker_emit_segment(tracker, tid, row, next_row, phase_row" in track_c_pass14,
            "pass 14 tracker must wire next_row into segment emission")


    estimator_h_pass15 = read("spectral_engine/analysis/spectral_peak_estimator.h")
    estimator_c_pass15 = read("spectral_engine/analysis/spectral_peak_estimator.c")
    config_pass15 = read("spectral_engine/core/spectral_config.h")
    track_h_pass15 = read("spectral_engine/analysis/spectral_peak_track.h")
    interp_c_pass15 = read("spectral_engine/analysis/spectral_peak_interp.c")
    track_c_pass15 = read("spectral_engine/analysis/spectral_peak_track.c")
    fused_c_pass15 = read("spectral_engine/analysis/spectral_analysis_fused.c")
    require("SPECTRAL_PEAK_PHASE_CONSISTENCY_TOL_RADS" in config_pass15,
            "pass 15 must define explicit phase-consistency tolerance")
    require("const float* next_phase_row;" in estimator_h_pass15 and
            "float phase_bin_offset;" in estimator_h_pass15 and
            "float phase_omega;" in estimator_h_pass15 and
            "float phase_error;" in estimator_h_pass15,
            "pass 15 estimator API must expose phase-advance diagnostics")
    require("SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID" in estimator_h_pass15 and
            "SPECTRAL_PEAK_ESTIMATE_PHASE_MODEL_CONSISTENT" in estimator_h_pass15,
            "pass 15 estimator flags must expose phase diagnostic state")
    require("spectral_peak_estimate_phase_advance" in estimator_c_pass15 and
            "phase_delta - model_omega * input->hop_float" in estimator_c_pass15,
            "pass 15 estimator must compute phase-advance model error")
    require("next_phase_row" in track_h_pass15 and
            "estimate_input.next_phase_row = next_phase_row;" in interp_c_pass15,
            "pass 15 tracker emission must pass next-frame phase into estimator")
    require("spectral_tracker_emit_segment(tracker, tid, row, next_row, phase_row, next_phase_row" in track_c_pass15,
            "pass 15 tracker must wire next_phase_row through segment emission")
    require("fctx.next_phase_row = phase_next;" in fused_c_pass15,
            "pass 15 fused analysis must supply next-frame phase row")


    estimator_h_pass16 = read("spectral_engine/analysis/spectral_peak_estimator.h")
    estimator_c_pass16 = read("spectral_engine/analysis/spectral_peak_estimator.c")
    track_h_pass16 = read("spectral_engine/analysis/spectral_peak_track.h")
    track_i_pass16 = read("spectral_engine/analysis/spectral_peak_track_internal.h")
    interp_c_pass16 = read("spectral_engine/analysis/spectral_peak_interp.c")
    track_c_pass16 = read("spectral_engine/analysis/spectral_peak_track.c")
    require("SpectralPeakPhasePolicy" in estimator_h_pass16 and
            "SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT" in estimator_h_pass16,
            "pass 16 estimator API must expose phase policy")
    require("phase_policy == SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT" in estimator_c_pass16,
            "pass 16 estimator must implement opt-in phase inconsistency rejection")
    require("spectral_tracker_set_phase_policy" in track_h_pass16 and
            "SpectralPeakPhasePolicy phase_policy;" in track_i_pass16,
            "pass 16 tracker must expose and store phase policy")
    require("SpectralPeakModel default_peak_model = spectral_peak_model_default();" in track_c_pass16 and
            "spectral_tracker_set_peak_model(tracker, &default_peak_model)" in track_c_pass16 and
            "model.phase_policy = policy;" in track_c_pass16,
            "pass 16 tracker must initialize phase policy through the resolved peak model")
    require("estimate_input.phase_policy = tracker->phase_policy;" in interp_c_pass16,
            "pass 16 tracker emission must pass phase policy into estimator")


    windows_h_pass17 = read("spectral_engine/core/spectral_windows.h")
    windows_c_pass17 = read("spectral_engine/core/spectral_windows.c")
    estimator_h_pass17 = read("spectral_engine/analysis/spectral_peak_estimator.h")
    estimator_c_pass17 = read("spectral_engine/analysis/spectral_peak_estimator.c")
    track_c_pass17 = read("spectral_engine/analysis/spectral_peak_track.c")
    require("SpectralWindowPeakMagsqFn" in windows_h_pass17 and
            "SpectralWindowPeakMagsqFn peak_magsq;" in windows_h_pass17,
            "pass 17 window descriptor must own peak-height estimation policy")
    require("spectral_window_peak_magsq_log_parabolic" in windows_c_pass17 and
            "spectral_window_peak_magsq_center" in windows_c_pass17,
            "pass 17 must provide log-parabolic and center-bin peak-height callbacks")
    require("SPECTRAL_WINDOW_RECTANGULAR" in windows_c_pass17 and
            "spectral_window_peak_magsq_center" in windows_c_pass17,
            "pass 17 rectangular window must not force log-parabolic amplitude correction")
    require("SpectralWindowPeakMagsqFn peak_magsq;" in estimator_h_pass17 and
            "spectral_peak_window_peak_magsq" in estimator_c_pass17,
            "pass 17 estimator must consume window peak-height callback")
    require("tracker->peak_magsq" in track_c_pass17,
            "pass 17 tracker must bind active window peak-height callback")


    peak_model_h_pass18 = read("spectral_engine/analysis/spectral_peak_model.h")
    peak_model_c_pass18 = read("spectral_engine/analysis/spectral_peak_model.c")
    manifest_pass18 = read("spectral_engine/cmake/source-manifest.cmake")
    track_h_pass18 = read("spectral_engine/analysis/spectral_peak_track.h")
    track_i_pass18 = read("spectral_engine/analysis/spectral_peak_track_internal.h")
    track_c_pass18 = read("spectral_engine/analysis/spectral_peak_track.c")
    require("spectral_peak_model.c" in manifest_pass18,
            "pass 18 peak model module must be compiled through source manifest")
    require("SpectralPeakModel" in peak_model_h_pass18 and
            "SpectralResolvedPeakModel" in peak_model_h_pass18 and
            "spectral_peak_model_validate" in peak_model_h_pass18,
            "pass 18 peak model API must expose validated and resolved model types")
    require("SPECTRAL_PEAK_MODEL_CAP_COMPLEX_TRIPLET" in peak_model_h_pass18 and
            "SPECTRAL_PEAK_MODEL_ASSUME_WINDOW_RECT_MAINLOBE" in peak_model_h_pass18,
            "pass 18 peak model must expose capability/assumption masks")
    require("spectral_peak_model_for_window_type" in peak_model_c_pass18 and
            "spectral_window_peak_magsq_log_parabolic" in peak_model_c_pass18,
            "pass 18 peak model implementation must validate window/amplitude coupling")
    require('#include "spectral_peak_model.h"' in track_h_pass18 and
            "spectral_tracker_set_peak_model" in track_h_pass18,
            "pass 18 tracker API must expose coherent peak-model setter")
    require("SpectralResolvedPeakModel peak_model;" in track_i_pass18,
            "pass 18 tracker internals must store one resolved peak model")
    require("spectral_tracker_apply_peak_model" in track_c_pass18 and
            "tracker->peak_model = *resolved;" in track_c_pass18 and
            "return spectral_tracker_apply_peak_model(tracker, model);" in track_c_pass18 and
            "fallback = spectral_peak_model_default()" not in track_c_pass18,
            "pass 18 tracker setters must resolve atomically and preserve prior models on failure")


    peak_model_h_pass20 = read("spectral_engine/analysis/spectral_peak_model.h")
    peak_model_c_pass20 = read("spectral_engine/analysis/spectral_peak_model.c")
    track_c_pass20 = read("spectral_engine/analysis/spectral_peak_track.c")
    require("spectral_peak_model_has_capability" in peak_model_h_pass20 and
            "spectral_peak_model_requires_phase_row" not in peak_model_h_pass20 and
            "spectral_peak_model_requires_next_phase_row" not in peak_model_h_pass20,
            "pass 20 peak model API must expose one generic capability helper, not alias-style requirement wrappers")
    require("int spectral_peak_model_has_capability" in peak_model_c_pass20 and
            "return (resolved.capabilities & capability) != 0u;" in peak_model_c_pass20,
            "pass 20 peak model capability checks must share one resolver")
    require("SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW |" in peak_model_c_pass20 and
            "SPECTRAL_PEAK_MODEL_CAP_NEXT_PHASE_ROW" in peak_model_c_pass20,
            "pass 20 phase diagnostics must require both current and next phase rows")
    require("return spectral_tracker_apply_peak_model(tracker, model);" in track_c_pass20,
            "pass 20 set_peak_model must remain transactional and not fallback to default")
    require("fallback = spectral_peak_model_default()" not in track_c_pass20,
            "pass 20 invalid model mutations must preserve existing profile, not fallback to Hann")


    oscillator_pass21 = read("spectral_engine/core/oscillator.c")
    synth_internal_pass21 = read("spectral_engine/core/spectral_synth_internal.h")
    synth_internal_c_pass21 = read("spectral_engine/core/spectral_synth_internal.c")
    segment_math_h_pass21 = read("spectral_engine/core/spectral_segment_math.h")
    segment_mt_pass21 = read("spectral_engine/core/spectral_segment_mt.c")
    spectral_in_pass21 = read("spectral_engine/core/spectral_in.c")
    spectral_out_pass21 = read("spectral_engine/core/spectral_out.c")
    processing_chain_pass21 = read("spectral_engine/analysis/spectral_processing_chain.c")
    synth_cpu_pass21 = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    synth_sim_pass21 = read("spectral_engine/synth/backends/sim/spectral_synth_simulation.c")
    track_c_pass21 = read("spectral_engine/analysis/spectral_peak_track.c")
    manifest_pass21 = read("spectral_engine/cmake/source-manifest.cmake")
    perf_pass21 = read("spectral_engine/runtime/spectral_perf.c")
    ai_canon_pass21 = read("docs/core_audit/AI_CANON.md")
    pass21_notes = read("docs/core_audit/PATCH_NOTES_PASS21.md")
    require("static inline float osc_sine" not in oscillator_pass21 and
            "static inline float osc_saw" not in oscillator_pass21 and
            "spectral_osc_sine," in oscillator_pass21 and
            "spectral_osc_pwm" in oscillator_pass21,
            "pass 21 oscillator dispatch must point at canonical formulas without local alias wrappers")
    require("compute_phase(" not in synth_internal_pass21 and
            "compute_amplitude(" not in synth_internal_pass21,
            "pass 21 synth internals must not re-alias canonical segment math helpers")
    require("spectral_segment_phase_at_index_f32" not in segment_math_h_pass21 and
            "spectral_segment_amp_at_index_f32" not in segment_math_h_pass21 and
            "spectral_segment_math.c" not in manifest_pass21,
            "pass 21 segment math must not keep unused out-of-line alias wrappers")
    require(perf_pass21.count("void perf_get_cpu_time") == 1,
            "pass 21 desktop CPU-time accounting must not be duplicated per platform branch")
    require(not (ROOT / "spectral_engine/analysis/spectral_peak_chain.c").exists() and
            not (ROOT / "spectral_engine/analysis/spectral_peak_chain.h").exists(),
            "pass 21 must remove the dead peak-chain files instead of preserving duplicate segment-copy logic")
    require("static void spectral_tracker_free_segment_storage" in track_c_pass21 and
            "spectral_calloc_array((size_t)n_threads, sizeof(TrackSegment*))" in track_c_pass21 and
            track_c_pass21.count("spectral_tracker_free_segment_storage(tracker);") >= 3,
            "pass 21 tracker ownership cleanup must use one helper and NULL-safe per-thread storage")
    require(synth_internal_c_pass21.count("lp.amp = s->amp;") == 1 and
            "static void gpu_tile_preprocess_scratch_free" in synth_internal_c_pass21 and
            synth_internal_c_pass21.count("gpu_tile_preprocess_scratch_free(") == 3,
            "pass 21 synth internals must remove repeated field assignment and share GPU tile scratch cleanup")
    require("static void segment_array_mt_apply_pending_locked" in segment_mt_pass21 and
            segment_mt_pass21.count("segment_array_mt_apply_pending_locked(sa);") == 2,
            "pass 21 segment MT get/copy must share one pending-swap commit helper")
    require("float* audio = malloc(" not in spectral_in_pass21 and
            "float* mono = malloc(" not in spectral_in_pass21 and
            "float* stereo = malloc(" not in spectral_out_pass21 and
            "spectral_size_add(n, 1u, &buf_len)" in processing_chain_pass21 and
            "tb.arena = spectral_calloc_array(arena_size, 1u)" in synth_cpu_pass21 and
            "memset(tb.arena" not in synth_cpu_pass21 and
            "sim_segs = (SimSegment*)spectral_malloc_array" in synth_sim_pass21 and
            "accum = (int64_t*)spectral_calloc_array" in synth_sim_pass21,
            "pass 21 array allocations must reuse safe allocation helpers in touched kernel paths")
    require("Alias wrappers are not architecture" in ai_canon_pass21 and
            "Do not add a wrapper whose only job is to rename another helper" in pass21_notes and
            "dead duplicate kernel files" in pass21_notes and
            "central cleanup helpers" in pass21_notes and
            "safe allocation helpers" in pass21_notes,
            "pass 21 docs must forbid low-value aliases, repeated cleanup and local allocation reinvention")


    synth_internal_pass22 = read("spectral_engine/core/spectral_synth_internal.c")
    synth_cpu_pass22 = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    require("out_len * elem_size overflow is a contract failure" in synth_internal_pass22,
            "pass 22 synth preflight must treat output byte-count overflow as an error, not benign early exit")
    require("pf.error = SPECTRAL_ERR_OVERFLOW;" in synth_internal_pass22 and
            "!spectral_size_mul(out_len, elem_size, &preflight_out_bytes)" in synth_internal_pass22,
            "pass 22 shared synth preflight must propagate SPECTRAL_ERR_OVERFLOW")
    require("memset(out_buffer, 0, out_bytes);" in synth_cpu_pass22 and
            "memset(out_buffer, 0, out_len * elem_size)" not in synth_cpu_pass22,
            "pass 22 CPU synth allocation-failure zeroing must use validated byte count")


    synth_cpu_pass23 = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    require("spectral_size_add(tb.buf_size, align_mask, &stride_with_padding)" in synth_cpu_pass23 and
            "tb.buf_stride = stride_with_padding & ~align_mask;" in synth_cpu_pass23,
            "pass 23 CPU synth arena stride alignment must use checked arithmetic")
    require("tb.buf_stride = (tb.buf_size + SPECTRAL_CACHE_ALIGN - 1)" not in synth_cpu_pass23,
            "pass 23 CPU synth arena stride must not use raw potentially-overflowing addition")
    require("base > UINTPTR_MAX - (uintptr_t)align_mask" in synth_cpu_pass23,
            "pass 23 CPU synth aligned-base pointer arithmetic must be guarded")
    require("thread_buffers_alloc(n_parts, out_len, elem_size, &tb_err)" in synth_cpu_pass23 and
            "return tb_err != SPECTRAL_OK ? tb_err : SPECTRAL_ERR_MEMORY;" in synth_cpu_pass23,
            "pass 23 CPU synth thread-buffer allocator must propagate overflow separately from memory failure")
    require("Checked align-up" in synth_cpu_pass23 and
            "Manual aligned-base adjustment" in synth_cpu_pass23 and
            "Pointer-add guard" in synth_cpu_pass23,
            "pass 23 source comments must explain checked align-up, arena padding and pointer-add guard")
    pass23_notes = read("docs/core_audit/PATCH_NOTES_PASS23.md")
    require("Reviewer Walkthrough" in pass23_notes and
            "checked align-up" in pass23_notes and
            "pointer-add guard" in pass23_notes and
            "SPECTRAL_ERR_OVERFLOW" in pass23_notes and
            "SPECTRAL_ERR_MEMORY" in pass23_notes,
            "pass 23 walkthrough must document overflow-vs-memory failure and the guarded arithmetic sequence")


    track_c_pass24 = read("spectral_engine/analysis/spectral_peak_track.c")
    interp_c_pass24 = read("spectral_engine/analysis/spectral_peak_interp.c")
    require("spectral_size_mul((size_t)n_threads, (size_t)SPECTRAL_CACHE_LINE_STRIDE, &thread_slots)" in track_c_pass24 and
            "spectral_size_mul(thread_slots, sizeof(size_t), &thread_slots_bytes)" in track_c_pass24,
            "pass 24 tracker create segment storage must check padded counter byte counts")
    require("spectral_size_mul(init_cap, sizeof(TrackSegment), &init_seg_bytes)" in track_c_pass24 and
            "spectral_aligned_alloc(init_seg_bytes)" in track_c_pass24,
            "pass 24 tracker create segment arrays must use checked TrackSegment byte counts")
    require("(size_t)n_threads * SPECTRAL_CACHE_LINE_STRIDE * sizeof(size_t)" not in track_c_pass24 and
            "init_cap * sizeof(TrackSegment)" not in track_c_pass24,
            "pass 24 tracker create must not use raw segment-storage allocation products")
    require("spectral_size_mul(new_cap, sizeof(TrackSegment), &new_bytes)" in interp_c_pass24 and
            "spectral_size_mul(count, sizeof(TrackSegment), &copy_bytes)" in interp_c_pass24,
            "pass 24 tracker segment growth must check allocation and copy byte counts")
    require("spectral_aligned_alloc(new_bytes)" in interp_c_pass24 and
            "memcpy(new_arr, tracker->seg_arrays[tid], copy_bytes)" in interp_c_pass24,
            "pass 24 tracker segment growth must use checked byte counts")


    track_internal_pass25 = read("spectral_engine/analysis/spectral_peak_track_internal.h")
    track_c_pass25 = read("spectral_engine/analysis/spectral_peak_track.c")
    interp_c_pass25 = read("spectral_engine/analysis/spectral_peak_interp.c")
    require("void spectral_tracker_set_error(SpectralTracker* tracker, SpectralError error);" in track_internal_pass25,
            "pass 25 tracker error propagation must expose internal first-error setter")
    require("atomic_compare_exchange_strong_explicit" in track_c_pass25 and
            "expected = SPECTRAL_OK" in track_c_pass25,
            "pass 25 tracker error propagation must preserve first non-OK error")
    require("spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);" in track_c_pass25 and
            "atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_MEMORY" not in track_c_pass25,
            "pass 25 tracker failure helper must not overwrite an existing root-cause error")
    require("spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);" in interp_c_pass25 and
            "spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);" in interp_c_pass25,
            "pass 25 emit path must record overflow/memory through first-error setter")


    seg_cache_pass26 = read("spectral_engine/core/spectral_seg_cache.c")
    require("spectral_size_add((size_t)count, 1u, &new_count)" in seg_cache_pass26 and
            "new_count > (size_t)UINT32_MAX" in seg_cache_pass26,
            "pass 26 segment cache index insertion must check persistent entry-count overflow")
    require("spectral_array_bytes(new_count, sizeof(SpectralSegCacheEntry), &new_entries_bytes)" in seg_cache_pass26,
            "pass 26 segment cache index insertion must check full index allocation byte count")
    require("spectral_array_bytes((size_t)ins, sizeof(SpectralSegCacheEntry), &prefix_bytes)" in seg_cache_pass26 and
            "spectral_array_bytes(suffix_count, sizeof(SpectralSegCacheEntry), &suffix_bytes)" in seg_cache_pass26,
            "pass 26 segment cache index insertion must check prefix/suffix copy byte counts")
    require("(size_t)(count + 1)" not in seg_cache_pass26 and
            "count++;" not in seg_cache_pass26,
            "pass 26 segment cache index insertion must not use wrapping uint32_t count increment")


    seg_cache_pass27 = read("spectral_engine/core/spectral_seg_cache.c")
    require("size_t seg_bytes = 0;" in seg_cache_pass27 and
            "size_t gpu_seg_bytes = 0;" in seg_cache_pass27 and
            "size_t tile_ranges_bytes = 0;" in seg_cache_pass27 and
            "size_t tile_refs_bytes = 0;" in seg_cache_pass27,
            "pass 27 segment cache append must precompute checked blob byte counts")
    require("spectral_seg_cache_fs_data_append_write(&w, scratch, seg_bytes)" in seg_cache_pass27 and
            "spectral_seg_cache_fs_data_append_write(&w, scratch, gpu_seg_bytes)" in seg_cache_pass27,
            "pass 27 segment cache append must use checked segment and GPU-segment byte counts")
    require("spectral_seg_cache_fs_data_append_write(&w, ranges_scratch, tile_ranges_bytes)" in seg_cache_pass27 and
            "spectral_seg_cache_fs_data_append_write(&w, refs_scratch, tile_refs_bytes)" in seg_cache_pass27,
            "pass 27 segment cache append must use checked tile scratch byte counts")
    require("(size_t)sa->count * sizeof(Segment)" not in seg_cache_pass27 and
            "(size_t)sa->count * sizeof(SegmentGpu)" not in seg_cache_pass27 and
            "(size_t)tile_count * sizeof(uint32_t) * 2u" not in seg_cache_pass27,
            "pass 27 segment cache append must not use raw blob byte products")
    require("uint32_t tc = has_tile_blob ? tile_count : 0;" in seg_cache_pass27 and
            "uint32_t tr = has_tile_blob ? tile_total_refs : 0;" in seg_cache_pass27,
            "pass 27 segment cache metadata must match whether tile bytes were written")


    seg_cache_fs_pass28 = read("spectral_engine/core/spectral_seg_cache_fs.c")
    require("spectral_size_add(sizeof(SpectralSegCacheHeader), entries_bytes, &expected_index_bytes)" in seg_cache_fs_pass28,
            "pass 28 segment cache index load must derive expected index file size with checked addition")
    require("spectral_fs_file_size(f, &file_size)" in seg_cache_fs_pass28 and
            "file_size != (uint64_t)expected_index_bytes" in seg_cache_fs_pass28,
            "pass 28 segment cache index load must validate count-derived file size before allocation")
    require("file_size != (uint64_t)sizeof(SpectralSegCacheHeader)" in seg_cache_fs_pass28,
            "pass 28 zero-count segment cache index must have exact header size")
    require(seg_cache_fs_pass28.index("file_size != (uint64_t)expected_index_bytes") <
            seg_cache_fs_pass28.index("entries = (SpectralSegCacheEntry*)spectral_malloc_array("),
            "pass 28 segment cache index load must not allocate before validating file shape")


    seg_cache_pass29 = read("spectral_engine/core/spectral_seg_cache.c")
    require("seg_cache_entry_metadata_valid" in seg_cache_pass29 and
            "e->sample_rate < (uint32_t)SPECTRAL_MIN_SAMPLE_RATE" in seg_cache_pass29 and
            "(uint64_t)(size_t)e->output_length != e->output_length" in seg_cache_pass29,
            "pass 29 segment cache lookup must validate scalar metadata before narrowing")
    require("if (!seg_cache_entry_metadata_valid(e))" in seg_cache_pass29 and
            seg_cache_pass29.index("if (!seg_cache_entry_metadata_valid(e))") <
            seg_cache_pass29.index("result->sample_rate = (int)e->sample_rate;"),
            "pass 29 segment cache lookup must validate metadata before writing public result")
    require("seg_cache_validate_data_extent" in seg_cache_pass29 and
            "spectral_seg_cache_fs_data_file_size(cache_dir, &data_file_size)" in seg_cache_pass29,
            "pass 29 segment cache lookup must validate declared data extent")
    require("err = seg_cache_validate_data_extent(cache_dir, e->data_offset, total_data_bytes);" in seg_cache_pass29 and
            seg_cache_pass29.index("err = seg_cache_validate_data_extent(cache_dir, e->data_offset, total_data_bytes);") <
            seg_cache_pass29.index("spectral_seg_cache_fs_data_map_ro("),
            "pass 29 segment cache lookup must validate data extent before mapping")


    seg_cache_pass30 = read("spectral_engine/core/spectral_seg_cache.c")
    require("seg_cache_scale_to_i32" in seg_cache_pass30 and
            "scaled < (double)INT_MIN" in seg_cache_pass30 and
            "scaled > (double)INT_MAX" in seg_cache_pass30,
            "pass 30 segment cache key must guard float-to-int conversions")
    require("(int)(db_thresh * 10.0f)" not in seg_cache_pass30 and
            "(int)(start_sec * 1000.0f)" not in seg_cache_pass30 and
            "(int)(end_sec * 1000.0f)" not in seg_cache_pass30 and
            "(int)(stretch * 1000000.0f)" not in seg_cache_pass30,
            "pass 30 segment cache key must not use raw float-to-int casts")
    require("len <= 0 || (size_t)len >= sizeof(buf)" in seg_cache_pass30 and
            "len = (int)(sizeof(buf) - 1)" not in seg_cache_pass30,
            "pass 30 segment cache key must reject truncated parameter strings")
    require("seg_cache_store_metadata_valid" in seg_cache_pass30 and
            "sample_rate < SPECTRAL_MIN_SAMPLE_RATE" in seg_cache_pass30 and
            "pitch > SPECTRAL_MAX_PITCH" in seg_cache_pass30,
            "pass 30 segment cache store must validate scalar metadata before persistence")
    require("if (!seg_cache_store_metadata_valid(key, sa, sample_rate, stretch, pitch, output_length))" in seg_cache_pass30,
            "pass 30 segment cache store must fail closed before opening append writer")


    pipeline_pass31 = read("spectral_engine/cmd/cli/spectral_cli_pipeline.c")
    seg_cache_h_pass31 = read("spectral_engine/core/spectral_seg_cache.h")
    seg_cache_c_pass31 = read("spectral_engine/core/spectral_seg_cache.c")
    require("pipeline_hash_input_file" in pipeline_pass31 and
            "spectral_hash_file_method_consume_file(&method, f)" in pipeline_pass31 and
            "spectral_hash_file_method_digest(&method, out_digest)" in pipeline_pass31,
            "pass 31 segment cache key must include input content digest")
    require('snprintf(input_id, sizeof(input_id), "%s|xxh=%016llx"' in pipeline_pass31 and
            "spectral_seg_cache_key(input_id, opts->n_fft, opts->hop" in pipeline_pass31,
            "pass 31 pipeline cache key must pass content-derived input identity")
    require("spectral_seg_cache_key(stem, opts->n_fft" not in pipeline_pass31,
            "pass 31 pipeline cache key must not be basename-only")
    require("const char* input_id" in seg_cache_h_pass31 and
            "spectral_is_empty_string(input_id)" in seg_cache_c_pass31,
            "pass 31 core segment cache key API must require non-empty input identity")


    hash_c_pass32 = read("spectral_engine/core/spectral_hash_xx32_xx3.c")
    require("uint64_t total_len_u64 = 0;" in hash_c_pass32 and
            "total_len_u64 > (uint64_t)SIZE_MAX" in hash_c_pass32 and
            "total_len = (size_t)total_len_u64;" in hash_c_pass32,
            "pass 32 hash full-direct must check uint64_t file length before narrowing to size_t")
    require("total_len = (size_t)(end_pos - start_pos)" not in hash_c_pass32,
            "pass 32 hash full-direct must not silently narrow file length")
    require("start_pos > (uint64_t)INT64_MAX" in hash_c_pass32,
            "pass 32 hash full-direct must guard uint64_t to int64_t seek position conversion")
    require("FULL_MMAP is registered for future support but is not advertised as available" in hash_c_pass32 and
            ".available = 0" in hash_c_pass32,
            "pass 32 unimplemented full_mmap hash method must not be advertised available")


    parser_pass33 = read("spectral_engine/core/spectral_segment_parser.c")
    require("segment_file_metadata_valid_u32" in parser_pass33 and
            "SPECTRAL_MIN_SAMPLE_RATE" in parser_pass33 and
            "SPECTRAL_MAX_STRETCH" in parser_pass33 and
            "SPECTRAL_MAX_PITCH" in parser_pass33,
            "pass 33 segment file parser must validate scalar metadata on save/load")
    require("segment_file_expected_bytes" in parser_pass33 and
            "spectral_fs_file_size(f, &file_size)" in parser_pass33 and
            "file_size != (uint64_t)expected_file_bytes" in parser_pass33,
            "pass 33 segment file parser must validate exact file shape before allocation")
    require("spectral_malloc_array((size_t)hdr.count, sizeof(Segment))" in parser_pass33 and
            "malloc(alloc_bytes)" not in parser_pass33,
            "pass 33 segment file load must use checked array allocator after file-shape validation")
    require("segments_validate_all(sa->segs, sa->count, &bad_idx)" in parser_pass33 and
            "Refusing to save corrupt segment data" in parser_pass33,
            "pass 33 segment file save must not persist corrupt segment arrays")


    wavetable_pass34 = read("spectral_engine/core/spectral_wavetable.c")
    require("wavetable_file_expected_bytes" in wavetable_pass34 and
            "spectral_fs_file_size(f, &file_size)" in wavetable_pass34 and
            "file_size != (uint64_t)expected_file_bytes" in wavetable_pass34,
            "pass 34 wavetable parser must validate exact .spwt file shape before payload read")
    require("wavetable_float_samples_finite" in wavetable_pass34 and
            "!spectral_is_finite_f32(samples[i])" in wavetable_pass34,
            "pass 34 wavetable parser must reject non-finite float samples")
    require("spectral_fs_read_exact(f, temp, payload_bytes, SPECTRAL_ERR_FILE_READ)" in wavetable_pass34,
            "pass 34 wavetable parser must use exact payload reads")
    require("spectral_fs_write_exact(f, &hdr, sizeof(hdr), SPECTRAL_ERR_FILE_WRITE)" in wavetable_pass34 and
            "spectral_fs_write_exact(f, table->samples, sample_bytes, SPECTRAL_ERR_FILE_WRITE)" in wavetable_pass34,
            "pass 34 wavetable save must use exact checked writes")


    wavetable_pass35 = read("spectral_engine/core/spectral_wavetable.c")
    require("file_size != (uint64_t)sample_bytes" in wavetable_pass35 and
            "spectral_fs_read_exact(f, temp, sample_bytes, SPECTRAL_ERR_FILE_READ)" in wavetable_pass35,
            "pass 35 wavetable raw loader must validate exact file size and use exact read")
    require("int saw_eof = 0;" in wavetable_pass35 and
            "if (!saw_eof)" in wavetable_pass35 and
            "covered_bytes != expected_bytes" in wavetable_pass35,
            "pass 35 wavetable HEX loader must require EOF record and full byte coverage")
    require("data_len > expected_bytes - offset" in wavetable_pass35 and
            "(size_t)address + data_len > expected_bytes" not in wavetable_pass35,
            "pass 35 wavetable HEX loader must use subtractive bounds checking")
    require("wavetable_runtime_samples_valid(temp, SPECTRAL_WAVETABLE_SIZE)" in wavetable_pass35 and
            "wavetable_runtime_samples_valid(temp_table, SPECTRAL_WAVETABLE_SIZE)" in wavetable_pass35 and
            "wavetable_runtime_samples_valid(data, SPECTRAL_WAVETABLE_SIZE)" in wavetable_pass35,
            "pass 35 wavetable raw/hex/buffer ingress must validate runtime samples before publishing")


    audio_in_pass36 = read("spectral_engine/core/spectral_in.c")
    audio_out_pass36 = read("spectral_engine/core/spectral_out.c")
    require("spectral_sf_count_to_size" in audio_in_pass36 and
            "if (!spectral_sf_count_to_size(sfinfo.frames, &frames))" in audio_in_pass36,
            "pass 36 audio IO must prove libsndfile input frame count is representable as size_t")
    require("spectral_size_mul(frames, channels, &total_samples)" in audio_in_pass36 and
            "memcpy(mono, audio, mono_bytes)" in audio_in_pass36 and
            "(size_t)sfinfo.frames * sizeof(float)" not in audio_in_pass36,
            "pass 36 audio IO input path must reuse checked frame byte counts")
    require("spectral_size_to_sf_count" in audio_out_pass36 and
            "if (!spectral_size_to_sf_count(num_frames, &frames_sf)" in audio_out_pass36,
            "pass 36 audio IO must prove output frame count is representable as sf_count_t")
    require("info.frames = frames_sf;" in audio_out_pass36 and
            "sf_writef_float(file, buffer, frames_sf)" in audio_out_pass36 and
            "(sf_count_t)num_frames" not in audio_out_pass36,
            "pass 36 audio IO output path must not silently narrow size_t frame counts")


    audio_in_pass37 = read("spectral_engine/core/spectral_in.c")
    audio_out_pass37 = read("spectral_engine/core/spectral_out.c")
    require("spectral_audio_samples_finite(audio, total_samples)" in audio_in_pass37 and
            audio_in_pass37.index("spectral_audio_samples_finite(audio, total_samples)") <
            audio_in_pass37.index("mono = (float*)spectral_malloc_array(frames, sizeof(float));"),
            "pass 37 audio IO input path must reject non-finite samples before publishing mono")
    require("const double inv_ch = 1.0 / (double)channels;" in audio_in_pass37 and
            "mono_d < -(double)FLT_MAX" in audio_in_pass37 and
            "mono_d > (double)FLT_MAX" in audio_in_pass37,
            "pass 37 audio IO downmix must prove finite float narrowing")
    require("spectral_audio_samples_finite(buffer, sample_count)" in audio_out_pass37 and
            audio_out_pass37.index("spectral_audio_samples_finite(buffer, sample_count)") <
            audio_out_pass37.index("file = sf_open(path, SFM_WRITE, &info);"),
            "pass 37 audio IO output path must reject non-finite samples before opening output")
    require("spectral_audio_samples_finite(mono, num_frames)" in audio_out_pass37,
            "pass 37 stereo writer must reject non-finite mono source before duplication")


    audio_out_pass38 = read("spectral_engine/core/spectral_out.c")
    require("spectral_wav_seek_u64_checked" in audio_out_pass38 and
            "offset > (uint64_t)INT64_MAX" in audio_out_pass38,
            "pass 38 WAV PEAK scrubber must reject unrepresentable seek offsets")
    require("spectral_fs_file_size(f, &file_size)" in audio_out_pass38 and
            "file_size - chunk_header_pos < (uint64_t)sizeof(chunk)" in audio_out_pass38,
            "pass 38 WAV PEAK scrubber must validate chunk header extent before reading")
    require("(uint64_t)size > file_size - data_pos" in audio_out_pass38 and
            "next_chunk_pos > file_size" in audio_out_pass38,
            "pass 38 WAV PEAK scrubber must validate RIFF chunk extents before seeking")
    require("spectral_fs_read_exact(f, chunk, sizeof(chunk), SPECTRAL_ERR_FILE_READ)" in audio_out_pass38 and
            "spectral_fs_write_exact(f, zero4, sizeof(zero4), SPECTRAL_ERR_FILE_WRITE)" in audio_out_pass38,
            "pass 38 WAV PEAK scrubber must use exact I/O for chunk headers and timestamp write")


    synth_internal_pass39 = read("spectral_engine/core/spectral_synth_internal.c")
    require("synth_derive_param_scalars" in synth_internal_pass39 and
            "stretch_sq = stretch * stretch;" in synth_internal_pass39 and
            "stretch_sq <= 0.0f" in synth_internal_pass39,
            "pass 39 synth params must reject tiny stretch values whose square underflows")
    require("inv_stretch = 1.0f / stretch;" in synth_internal_pass39 and
            "inv_stretch_sq = 1.0f / stretch_sq;" in synth_internal_pass39 and
            "spectral_is_finite_positive_f32(inv_stretch_sq)" in synth_internal_pass39,
            "pass 39 synth params must validate finite derived reciprocal scalars")
    require("synth_derive_param_scalars(stretch, pitch," in synth_internal_pass39 and
            ".inv_stretch = inv_stretch" in synth_internal_pass39 and
            ".inv_stretch_sq = inv_stretch_sq" in synth_internal_pass39,
            "pass 39 make_synth_params must reuse checked derived scalars")
    require(".inv_stretch_sq = 1.0f / (stretch * stretch)" not in synth_internal_pass39,
            "pass 39 make_synth_params must not recompute unchecked reciprocal-square stretch")


    seg_cache_pass40 = read("spectral_engine/core/spectral_seg_cache.c")
    require("seg_cache_segment_valid" in seg_cache_pass40 and
            "!spectral_is_finite_f32(s->omega) || s->omega < 0.0f" in seg_cache_pass40,
            "pass 40 segment cache payload must validate Segment fields")
    require("seg_cache_gpu_segments_match" in seg_cache_pass40 and
            "SegmentGpu expected = spectral_segment_pack_gpu(&segs[i]);" in seg_cache_pass40,
            "pass 40 segment cache payload must validate mmap GPU-segment payloads against Segment payloads")
    require("seg_cache_gpu_segments_match(mapped_segments, mapped_gpu_segs" in seg_cache_pass40 and
            "spectral_seg_cache_fs_data_unmap(&mv)" in seg_cache_pass40,
            "pass 40 segment cache lookup must reject corrupt mmap payloads before publishing result")
    require("if (!seg_cache_segments_valid(segs, e->seg_count, NULL))" in seg_cache_pass40,
            "pass 40 segment cache heap fallback must validate copied Segment payload")
    require("if (!seg_cache_segments_valid(sa->segs, sa->count, NULL))" in seg_cache_pass40 and
            "!seg_cache_tile_layout_words_valid(tile_ranges," in seg_cache_pass40,
            "pass 40 segment cache store must reject corrupt Segment/tile payloads before append")


    synth_internal_pass41 = read("spectral_engine/core/spectral_synth_internal.c")
    require("!spectral_is_finite_f32(alpha)" in synth_internal_pass41 and
            "!spectral_is_finite_f32(beta)" in synth_internal_pass41 and
            "!spectral_is_finite_f32(d_amp)" in synth_internal_pass41,
            "pass 41 segment loop must reject non-finite derived hot-loop scalars")
    require("spectral_segment_phase_at_f32(s->phase, alpha, beta, last)" in synth_internal_pass41 and
            "spectral_segment_amp_at_f32(s->amp, d_amp, last)" in synth_internal_pass41,
            "pass 41 segment loop must validate endpoint phase/amplitude before marking valid")
    require("lp.length == 0" in synth_internal_pass41,
            "pass 41 segment loop must reject zero-length post-stretch segments")


    synth_header_pass42 = read("spectral_engine/core/spectral_synth_internal.h")
    metal_pass42 = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    cuda_pass42 = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")
    require("gpu_synth_params_pack_checked" in synth_header_pass42 and
            "sp->out_len > (size_t)UINT32_MAX" in synth_header_pass42,
            "pass 42 GPU synth params must expose checked size_t-to-uint32 ABI pack")
    require("gpu_synth_params_pack_checked(&pf.params, SPECTRAL_GPU_TILE_SIZE, timbre, &params)" in metal_pass42,
            "pass 42 Metal backend must use checked GPU params pack")
    require("gpu_synth_params_pack_checked(&pf.params, SPECTRAL_GPU_TILE_SIZE, timbre, &gp)" in cuda_pass42,
            "pass 42 CUDA backend must use checked GPU params pack")


    cuda_pass43 = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")
    require("cudaEvent_t ev_start = NULL;" in cuda_pass43 and
            "cudaEvent_t ev_stop = NULL;" in cuda_pass43,
            "pass 43 CUDA events must be in cleanup scope")
    require("cudaEventCreate(&ev_start) != cudaSuccess" in cuda_pass43 and
            "cudaEventCreate(&ev_stop) != cudaSuccess" in cuda_pass43,
            "pass 43 CUDA event creation must be checked")
    require("if (ev_start) cudaEventDestroy(ev_start);" in cuda_pass43 and
            "if (ev_stop) cudaEventDestroy(ev_stop);" in cuda_pass43,
            "pass 43 CUDA events must be destroyed on common cleanup path")


    metal_pass44 = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    require("if (!paramsBuffer)" in metal_pass44 and "if (!cmdBuffer)" in metal_pass44 and "if (!encoder)" in metal_pass44,
            "pass 44 Metal dispatch must validate dispatch objects before use")
    require("[cmdBuffer status] != MTLCommandBufferStatusCompleted" in metal_pass44,
            "pass 44 Metal dispatch must check command buffer completion before output copyback")
    require(metal_pass44.index("[cmdBuffer status] != MTLCommandBufferStatusCompleted") <
            metal_pass44.index("memcpy(out_buffer, [g_mtl.outputBuf contents], output_size)"),
            "pass 44 Metal dispatch must not copy stale output before status check")


    synth_internal_pass45 = read("spectral_engine/core/spectral_synth_internal.c")
    require("int fill_overflow = 0;" in synth_internal_pass45 and
            "pos >= tile_ranges[tt].count" in synth_internal_pass45 and
            "write_index >= total_refs" in synth_internal_pass45,
            "pass 45 GPU tile fill must bound-check atomic cursor writes")
    require("tile_cursors[t] != tile_counts[t]" in synth_internal_pass45,
            "pass 45 GPU tile fill must verify count/fill cursor agreement")
    require("#pragma omp atomic write" in synth_internal_pass45,
            "pass 45 GPU tile fill overflow flag must be set safely under OpenMP")


    cuda_pass46 = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")
    require("copy_err = cudaMemcpyAsync" in cuda_pass46 and "copy_err != cudaSuccess" in cuda_pass46,
            "pass 46 CUDA async copies must be checked")
    require("cudaEventRecord(ev_start, g_cuda.stream) != cudaSuccess" in cuda_pass46 and
            "cudaEventRecord(ev_stop, g_cuda.stream) != cudaSuccess" in cuda_pass46,
            "pass 46 CUDA event records must be checked")
    require("sync_err = cudaStreamSynchronize(g_cuda.stream)" in cuda_pass46 and
            "sync_err != cudaSuccess" in cuda_pass46,
            "pass 46 CUDA stream synchronization must be checked")
    require("cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop) != cudaSuccess" in cuda_pass46,
            "pass 46 CUDA timing query must be checked")


    synth_internal_pass47 = read("spectral_engine/core/spectral_synth_internal.c")
    require("last_offset_d = (double)(lp.length - 1u);" in synth_internal_pass47 and
            "last_offset_d > (double)FLT_MAX" in synth_internal_pass47,
            "pass 47 segment loop must prove sample offset is representable as float")
    require("const float last = (float)last_offset_d;" in synth_internal_pass47 and
            "const float last = (float)(lp.length - 1u);" not in synth_internal_pass47,
            "pass 47 segment loop endpoint validation must narrow only checked offsets")


    synth_internal_pass48 = read("spectral_engine/core/spectral_synth_internal.c")
    require("if (sa.count > (size_t)UINT32_MAX) return SPECTRAL_ERR_OVERFLOW;" in synth_internal_pass48,
            "pass 48 GPU tile preprocess must reject segment counts outside uint32 ID domain")
    require(synth_internal_pass48.index("if (sa.count > (size_t)UINT32_MAX) return SPECTRAL_ERR_OVERFLOW;") <
            synth_internal_pass48.index("#pragma omp parallel"),
            "pass 48 GPU tile preprocess must check segment-ID narrowing before parallel passes")


    metal_pass49 = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    cuda_pass49 = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")
    require("if (td.total_refs == 0u)" in metal_pass49 and
            "memset(out_buffer, 0, output_size);" in metal_pass49,
            "pass 49 zero-reference GPU tile layout must produce silence in Metal before dispatch")
    require("if (td.total_refs == 0u)" in cuda_pass49 and
            "memset(out_buffer, 0, out_size);" in cuda_pass49,
            "pass 49 zero-reference GPU tile layout must produce silence in CUDA before dispatch")


    synth_internal_pass50 = read("spectral_engine/core/spectral_synth_internal.c")
    require("gpu_tile_data_refs_valid" in synth_internal_pass50 and
            "segment_ids[i] >= segment_count" in synth_internal_pass50,
            "pass 50 GPU tile cache must validate cached segment references")
    require("if (num_tiles == 0u || (total_refs > 0u && (!ranges || !segment_ids)))" in synth_internal_pass50,
            "pass 50 GPU tile cache set must fail closed on invalid pointer/count shapes")
    require("gpu_tile_data_refs_valid(out_td->ranges" in synth_internal_pass50 and
            "gpu_tile_cache_clear();" in synth_internal_pass50,
            "pass 50 GPU tile cache hits must be validated before reuse")


    cpu_pass51 = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    require("memcpy(out_buffer, tb->bufs[0], out_bytes)" in cpu_pass51 and
            "out_len * sizeof(float)" not in cpu_pass51,
            "pass 51 CPU synth reduction must reuse checked byte count")
    require("typedef SpectralError (*ReduceFn)(const ThreadBuffers* tb, void* out_buffer, size_t out_len, size_t out_bytes);" in cpu_pass51,
            "pass 51 CPU synth reduction function signature must carry checked byte count")
    require("reduce_err = reduce_fn(&tb, out_buffer, out_len, out_bytes);" in cpu_pass51 and
            "return reduce_err;" in cpu_pass51,
            "pass 51 CPU synth driver must propagate reduction contract failures")


    out_pass52 = read("spectral_engine/core/spectral_out.c")
    require("spectral_float_buffer_all_finite" in out_pass52 and
            "!spectral_is_finite_f32(headroom) || headroom < 0.0f" in out_pass52,
            "pass 52 float normalization must validate headroom and input finiteness")
    require("if (!spectral_is_finite_f32(scale)) return 0.0f;" in out_pass52,
            "pass 52 float normalization must validate scale before applying")

    if FAILURES:
        for f in FAILURES:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("core static audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
