/* spectral_synth_simulation.c - Desktop simulation of embedded Q15 synthesis
 * Allows testing embedded synthesis path without hardware.
 */

#include "spectral_synth_arm32.h"
#include "spectral_synth_internal.h"
#include "spectral_error.h"
#include "spectral_common.h"
#include "spectral_q15.h"
#include "spectral_lut.h"
#include "spectral_wavetable.h"
#include "spectral_perf.h"
#include "spectral_perf_accounting.h"
#include "spectral_perf_model.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include "oscillator.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if !SPECTRAL_EMBEDDED || !SPECTRAL_EMBEDDED_SIMULATION
#error "spectral_synth_simulation.c requires SPECTRAL_EMBEDDED=1 and SPECTRAL_EMBEDDED_SIMULATION=1"
#endif

/* Embedded simulation configuration */

static EmbeddedTargetConfig g_sim_config;
static int g_sim_config_initialized = 0;

static SpectralPerfProfileId parse_perf_profile_env(const char* value) {
    if (spectral_is_empty_string(value)) return SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST;
    if (strcmp(value, "m7") == 0 || strcmp(value, "generic") == 0 ||
        strcmp(value, "m7_generic_worst") == 0) {
        return SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST;
    }
    if (strcmp(value, "daisy") == 0 || strcmp(value, "h750") == 0 ||
        strcmp(value, "daisy_h750_worst") == 0) {
        return SPECTRAL_PERF_PROFILE_DAISY_H750_WORST;
    }
    return SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST;
}

static void apply_perf_overrides_from_env(EmbeddedTargetConfig* cfg) {
    const char* profile_env = spectral_getenv_nonempty(SPECTRAL_ENV_SIM_PERF_PROFILE);
    double pessimism = 0.0;
    int cold_start = 0;

    if (!cfg) return;

    if (profile_env) {
        cfg->perf_profile = (uint32_t)parse_perf_profile_env(profile_env);
    }
    if (spectral_getenv_f64_positive(SPECTRAL_ENV_SIM_PESSIMISM, &pessimism)) {
        cfg->pessimism_override = pessimism;
    }
    if (spectral_getenv_bool(SPECTRAL_ENV_SIM_PERF_COLD, &cold_start)) {
        cfg->include_cold_start = cold_start;
    }
}

static EmbeddedTargetConfig* get_simulation_config(void) {
    if (!g_sim_config_initialized) {
        g_sim_config = embedded_perf_default_config();
        apply_perf_overrides_from_env(&g_sim_config);
        g_sim_config_initialized = 1;
    }
    return &g_sim_config;
}

void embedded_sim_set_config(uint32_t cpu_mhz, uint32_t sample_rate,
                             uint32_t block_size, uint32_t max_mem_kb) {
    EmbeddedTargetConfig* cfg = get_simulation_config();
    if (cpu_mhz > 0) cfg->cpu_freq_mhz = cpu_mhz;
    if (sample_rate > 0) cfg->sample_rate = sample_rate;
    if (block_size > 0) cfg->block_size = block_size;
    if (max_mem_kb > 0) cfg->max_memory_kb = max_mem_kb;
}

void embedded_sim_set_verbose(int verbose) {
    get_simulation_config()->verbose = verbose;
}

void embedded_sim_set_perf_profile(uint32_t profile_id) {
    EmbeddedTargetConfig* cfg = get_simulation_config();
    if (profile_id < (uint32_t)SPECTRAL_PERF_PROFILE_COUNT) {
        cfg->perf_profile = profile_id;
    }
}

void embedded_sim_set_pessimism(double factor) {
    if (!spectral_is_finite_positive_f64(factor)) return;
    get_simulation_config()->pessimism_override = factor;
}

void embedded_sim_set_cold_start_reporting(int enabled) {
    get_simulation_config()->include_cold_start = enabled ? 1 : 0;
}

/* Float Segment -> embedded SpectralSegmentQ15 conversion.
 *
 * This is the desktop-side equivalent of cmd/convert_segments.c, but it also
 * folds in the runtime stretch/pitch parameters because the real ARM32 backend
 * applies neither at synthesis time (spectral_arm32_set_stretch() is a no-op;
 * pitch/stretch are baked into start/length/frequency here):
 *   - start/length  : scaled by stretch (length clamped to the 16-bit field)
 *   - freq_q88       : omega*pitch*inv_stretch encoded as Q8.8 via OMEGA_TO_Q88
 *   - amp_q15/da_q15 : scaled by amp_scale (Q15 headroom normalization)
 *   - df_q15         : forced to 0 — the ARM32 hot path does not consume chirp
 *                      and spectral_arm32_load() rejects df_q15 != 0.
 * Returns 1 on a valid segment, 0 if the segment should be dropped. */
static int segment_to_q15(const Segment* src, SpectralSegmentQ15* dst,
                          float amp_scale, const SynthParams* params, size_t out_len) {
    double start_d = 0.0;
    double length_d = 0.0;
    float amp_scaled = 0.0f;
    float da_scaled = 0.0f;

    if (!src || !dst || !params || !spectral_segment_valid_for_synth(src) ||
        !spectral_is_finite_positive_f32(amp_scale) || out_len == 0u) {
        return 0;
    }
    *dst = (SpectralSegmentQ15){0};

    start_d = (double)src->start * (double)params->stretch;
    length_d = (double)src->length * (double)params->stretch;
    if (!spectral_is_finite_f64(start_d) || !spectral_is_finite_f64(length_d) ||
        start_d < 0.0 || start_d >= (double)out_len || start_d > (double)UINT32_MAX ||
        length_d <= 0.0) {
        return 0;
    }
    if (length_d > 65535.0) length_d = 65535.0;

    dst->start = (uint32_t)start_d;
    dst->length = (uint16_t)length_d;
    if (dst->length == 0u) return 0;

    dst->freq_q88 = OMEGA_TO_Q88(
        spectral_segment_alpha_f32(src->omega, params->pitch_factor, params->inv_stretch));
    dst->phase_q15 = PHASE_RAD_TO_Q15(src->phase);

    amp_scaled = spectral_clamp_f32(src->amp * amp_scale, 0.0f, 1.0f);
    dst->amp_q15 = FLOAT_TO_Q15(amp_scaled);

    da_scaled = spectral_segment_d_amp_f32(src->da, params->inv_stretch) * amp_scale;
    dst->da_q15 = FLOAT_TO_Q15(spectral_clamp_f32(da_scaled, -1.0f, 1.0f));
    /* df_q15 stays 0 (zero-initialized above): chirp is intentionally dropped. */
    return 1;
}

/* Oscillator LUT — supplied to the real ARM32 context as ctx->osc_lut. */

static const q15_t* get_simulation_lut(void) {
    static q15_t lut[SPECTRAL_OSC_LUT_SIZE + 1];
    static int initialized = 0;

    if (!initialized) {
        spectral_lut_init_sine(lut);
        initialized = 1;
    }
    return lut;
}

enum { SIMULATION_MAX_ACTIVE = SPECTRAL_ARM32_MAX_ACTIVE };

static SpectralError simulation_timbre_fallback_invoke(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    int n_threads, double* t_synth, void* user_data)
{
    (void)user_data;
    return synth_arm32_simulation(sa, out_buffer, out_len, stretch, pitch,
                                  timbre, n_threads, t_synth);
}

/* Desktop simulation entry point */

SpectralError synth_arm32_simulation(SegmentArray sa, float* out_buffer, size_t out_len,
                                     float stretch, float pitch, SpectralTimbre timbre,
                                     int n_threads, double* t_synth) {
    SpectralError result = SPECTRAL_OK;
    SpectralSegmentQ15* q15_src = NULL;   /* converted segments validated by load */
    SpectralSegmentQ15* q15_ctx = NULL;   /* working copy owned by the ARM32 ctx */
    SpectralArm32Ctx* ctx = NULL;
    uint32_t* active_idx = NULL;          /* workload-model active set (indices) */
    int continue_backend = 1;
    
    SynthPreflight pf = synth_preflight_float(out_buffer, out_len, sa, stretch, pitch, &t_synth);
    if (!pf.ok) {
        return pf.error;
    }
    if (out_len > (size_t)UINT32_MAX) {
        memset(out_buffer, 0, out_len * sizeof(float));
        if (t_synth) *t_synth = 0;
        return SPECTRAL_ERR_OVERFLOW;
    }

    result = spectral_handle_unsupported_timbre(
            SPECTRAL_RESOLUTION_BACKEND_SIMULATION,
            TIMBRE_SINE,
            timbre,
            TIMBRE_SINE,
            SPECTRAL_RESOLUTION_REASON_SIM_TIMBRE_LIMIT,
            sa, out_buffer, out_len,
            stretch, pitch, n_threads, t_synth,
            simulation_timbre_fallback_invoke, NULL, &continue_backend);
    if (result != SPECTRAL_OK) {
        return result;
    }
    if (!continue_backend) {
        return SPECTRAL_OK;
    }

    (void)n_threads;

    double start_time = spectral_get_time_sec();
    
    const SynthParams params = pf.params;
    EmbeddedTargetConfig* cfg = get_simulation_config();
    
    /* Find maximum amplitude for scaling all segments to fit in Q15 */
    float max_amp = 0.0f;
    for (size_t i = 0; i < sa.count; i++) {
        if (sa.segs[i].amp > max_amp) max_amp = sa.segs[i].amp;
    }
    /* amp_scale brings max_amp to 1.0, with a bit of headroom */
    float amp_scale = (max_amp > 0.0f) ? (SPECTRAL_SIMULATION_HEADROOM / max_amp) : 1.0f;
    
    /* Convert float segments to embedded Q15 form, dropping any that fail
     * conversion so the strict ARM32 loader sees only loadable data. The array
     * is compacted (invalid entries removed), preserving start ordering. */
    uint32_t loaded = 0;
    if (sa.count > 0) {
        q15_src = (SpectralSegmentQ15*)spectral_malloc_array(sa.count, sizeof(*q15_src));
        if (!q15_src) {
            spectral_log_warn_codef(SPECTRAL_ERROR_DOMAIN_CORE, SPECTRAL_ERR_MEMORY,
                                    "Simulation segment buffer allocation failed (segments=%zu)",
                                    sa.count);
            result = SPECTRAL_ERR_MEMORY;
            goto cleanup;
        }
        for (size_t i = 0; i < sa.count; i++) {
            SpectralSegmentQ15 tmp;
            if (segment_to_q15(&sa.segs[i], &tmp, amp_scale, &params, out_len)) {
                q15_src[loaded++] = tmp;
            }
        }
    }

    const SpectralPerfModelProfile* perf_profile =
        spectral_perf_model_profile((SpectralPerfProfileId)cfg->perf_profile);
    const uint32_t cache_miss_threshold =
        perf_profile ? perf_profile->cache_miss_threshold_active : 24u;
    if (!perf_profile) perf_profile = spectral_perf_model_default_profile();

#if SPECTRAL_DEBUG && !defined(NDEBUG)
    if (cfg->verbose) {
        SPECTRAL_DBG("max_amp=%.3f amp_scale=%.6f loaded=%u/%zu",
                     (double)max_amp, (double)amp_scale, loaded, sa.count);
    }
#endif

    /* Stand up the REAL ARM32 synthesis context. It owns a separate working copy
     * of the segments (spectral_arm32_load() validates ordering / active-count /
     * chirp constraints, then memcpy's into ctx->segments). All audio below comes
     * from spectral_arm32_process() — there is no parallel oscillator anymore. */
    const q15_t* osc_lut = get_simulation_lut();
    const uint32_t block_size = cfg->block_size;

    ctx = (SpectralArm32Ctx*)spectral_malloc_array(1u, sizeof(*ctx));
    if (!ctx) {
        result = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }
    if (loaded > 0) {
        q15_ctx = (SpectralSegmentQ15*)spectral_malloc_array(loaded, sizeof(*q15_ctx));
        if (!q15_ctx) {
            result = SPECTRAL_ERR_MEMORY;
            goto cleanup;
        }
    }
    spectral_arm32_init(ctx, q15_ctx, loaded, osc_lut, cfg->sample_rate);
    {
        SpectralError lerr = spectral_arm32_load(ctx, q15_src, loaded, (uint32_t)out_len);
        if (lerr != SPECTRAL_OK) {
            spectral_log_warn_codef(SPECTRAL_ERROR_DOMAIN_CORE, lerr,
                "ARM32 load rejected %u segment(s) (monotonic/active/chirp bound)", loaded);
            result = lerr;
            goto cleanup;
        }
    }

    /* Workload-accounting model (MEASURED side). Walk the same per-block segment
     * schedule that spectral_arm32_process() follows and tally the work — segment
     * activations, scan length, peak active set, and the per-sample LUT / MAC /
     * phase op counts that feed the cycle estimate. No samples are produced here;
     * this populates EmbeddedOpCounts only. */
    active_idx = (uint32_t*)spectral_malloc_array(SIMULATION_MAX_ACTIVE, sizeof(*active_idx));
    if (!active_idx) {
        result = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }

    EmbeddedOpCounts ops;
    spectral_perf_counters_reset(&ops);
    uint32_t num_active = 0;
    uint32_t peak_active = 0;
    uint32_t next_seg_idx = 0;

    for (size_t out_pos = 0; out_pos < out_len; out_pos += block_size) {
        uint32_t block_len = (out_len - out_pos > block_size)
            ? block_size : (uint32_t)(out_len - out_pos);
        uint32_t block_end = (uint32_t)out_pos + block_len;

        /* Activate segments starting within this block. */
        uint32_t block_activations = 0;
        uint32_t scan_start_idx = next_seg_idx;
        while (next_seg_idx < loaded && num_active < SIMULATION_MAX_ACTIVE) {
            const SpectralSegmentQ15* seg = &q15_src[next_seg_idx];
            if (seg->start >= block_end) break;
            uint32_t seg_end = seg->start + seg->length;
            if (seg_end > out_pos) {
                active_idx[num_active++] = next_seg_idx;
                block_activations++;
            }
            next_seg_idx++;
        }

        spectral_perf_count_segment_activations(&ops, block_activations);
        spectral_perf_count_segment_scan(&ops, (uint32_t)(next_seg_idx - scan_start_idx));
        if (num_active > peak_active) peak_active = num_active;
        spectral_perf_count_cache_pressure(&ops, num_active, cache_miss_threshold, block_len);

        uint64_t block_lut_lookups = 0;
        uint64_t block_mac_operations = 0;
        uint64_t block_phase_updates = 0;
        uint64_t block_loop_iterations = 0;

        uint32_t i = 0;
        while (i < num_active) {
            const SpectralSegmentQ15* seg = &q15_src[active_idx[i]];
            uint32_t seg_end = seg->start + seg->length;
            if (out_pos >= seg_end) {
                active_idx[i] = active_idx[--num_active];
                continue;
            }
            uint32_t blk_start = (seg->start > out_pos) ? (seg->start - (uint32_t)out_pos) : 0;
            uint32_t blk_end = (seg_end < block_end) ? (seg_end - (uint32_t)out_pos) : block_len;
            uint32_t len = blk_end - blk_start;

            spectral_perf_count_segment_samples(&ops, len);
            block_lut_lookups += len;
            block_mac_operations += len;
            block_phase_updates += len;
            block_loop_iterations += spectral_perf_loop_iters_for_samples(len);
            i++;
        }

        uint64_t block_cycles = spectral_perf_model_estimate_block_cycles(
            perf_profile, block_len, num_active, block_activations,
            (uint32_t)(next_seg_idx - scan_start_idx),
            block_lut_lookups, block_mac_operations,
            block_phase_updates, block_loop_iterations);
        spectral_perf_record_peak_block(&ops, block_cycles, num_active);
    }

#if SPECTRAL_DEBUG && !defined(NDEBUG)
    if (cfg->verbose) {
        SPECTRAL_DBG("peak_active=%u (max allowed=%d)", peak_active, SIMULATION_MAX_ACTIVE);
    }
#endif

    /* Audio generation (the real code). Drive spectral_arm32_process() across the
     * whole buffer in <=256-sample chunks (its hard block cap) and widen each Q15
     * result to float. With no segments loaded the process zeroes every chunk and
     * returns 0, so 'want' still advances the cursor (no infinite loop). */
    {
        q15_t qblk[256];
        size_t pos = 0;
        while (pos < out_len) {
            uint32_t want = (out_len - pos > 256u) ? 256u : (uint32_t)(out_len - pos);
            uint32_t got = spectral_arm32_process(ctx, qblk, NULL, want);
            uint32_t n = (got > 0u) ? got : want;
            for (uint32_t j = 0; j < n; j++) {
                out_buffer[pos + j] = Q15_TO_FLOAT(qblk[j]);
            }
            pos += n;
        }
    }

cleanup:
    free(active_idx);
    free(ctx);
    free(q15_ctx);
    free(q15_src);

    if (result != SPECTRAL_OK) {
        memset(out_buffer, 0, out_len * sizeof(float));
        if (t_synth) *t_synth = 0;
        return result;
    }

    {
        double elapsed = spectral_get_time_sec() - start_time;
        *t_synth = elapsed;

        /* Embedded-target perf/memory estimate over the loaded segment set. */
        EmbeddedPerfEstimate est = embedded_perf_estimate(
            cfg, &ops, out_len, loaded, peak_active, elapsed);
        embedded_perf_print(cfg, &est);

        EmbeddedMemoryUsage mem = embedded_memory_usage(
            loaded,
            block_size,
            SPECTRAL_OSC_LUT_BITS,
            SIMULATION_MAX_ACTIVE,
            cfg->max_memory_kb
        );
        embedded_memory_print(&mem);
    }

    return SPECTRAL_OK;
}

/* synth_cpu is provided via macro in spectral_synth_arm32.h
 * when SPECTRAL_USE_EMBEDDED_SYNTH is defined, redirecting to synth_arm32_simulation */

/* Wavetable version - falls back to simulation (wavetables not yet supported) */
#ifdef SPECTRAL_USE_EMBEDDED_SYNTH
SpectralError synth_cpu_wavetable(SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch,
                                  const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                  int n_threads, double* t_synth) {
    (void)bank;
    if (bank != NULL) {
        char resolution[384] = {0};
        SpectralResolutionContext context = {
            .event = SPECTRAL_RESOLUTION_EVENT_FALLBACK,
            .backend = SPECTRAL_RESOLUTION_BACKEND_SIMULATION,
            .scope = SPECTRAL_RESOLUTION_SCOPE_WAVETABLE,
            .requested_label = "enabled",
            .requested_id = 1,
            .effective_label = "disabled",
            .effective_id = 0,
            .max_supported_id = -1,
            .mode = spectral_exec_mode_name(),
            .reason = SPECTRAL_RESOLUTION_REASON_SIM_WAVETABLE_UNSUPPORTED
        };
        spectral_format_resolution_context(resolution, sizeof(resolution), &context);
        SPECTRAL_WARN_ONCE(TIMBRE_COUNT + 1, "%s", resolution);
    }
    return synth_arm32_simulation(sa, out_buffer, out_len, stretch, pitch,
                                  timbre, n_threads, t_synth);
}
#endif
