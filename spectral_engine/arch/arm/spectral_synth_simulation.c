/* spectral_synth_simulation.c - Desktop simulation of embedded Q15 synthesis
 * Allows testing embedded synthesis path without hardware.
 */

#include "spectral_synth.h"   /* the synth_cpu contract this TU implements */
#include "spectral_synth_arm32.h"
#include "spectral_synth_internal.h"
#include "spectral_error.h"
#include "spectral_common.h"
#include "spectral_q.h"
#include "spectral_lut.h"
#include "spectral_wavetable.h"
#include "spectral_perf.h"
#include "spectral_perf_accounting.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include "spectral_segment_convert.h"
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

/* Last synthesis run's embedded-target report, for the caller to print. */
static struct {
    EmbeddedTargetConfig cfg;
    EmbeddedPerfEstimate est;
    EmbeddedMemoryUsage  mem;
    int valid;
} g_last_report;

static EmbeddedTargetConfig* get_simulation_config(void) {
    if (!g_sim_config_initialized) {
        g_sim_config = embedded_perf_default_config();
        g_sim_config_initialized = 1;
    }
    return &g_sim_config;
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
    return synth_cpu(sa, out_buffer, out_len, stretch, pitch,
                     timbre, n_threads, t_synth);
}

/* Desktop simulation entry point */

/* The build-selected CPU synthesis entry: in the embedded-synth profiles
 * THIS TU defines synth_cpu (the float OpenMP body in spectral_synth_cpu.c
 * compiles itself out), so callers dispatch by symbol — no rename macro. */
SpectralError synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len,
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
            if (spectral_segment_to_q15_runtime(&sa.segs[i], &tmp, amp_scale, &params, out_len)) {
                q15_src[loaded++] = tmp;
            }
        }
    }

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
     * activations, scan length, peak active set, and per-voice sample counts.
     * No samples are produced here; this populates EmbeddedOpCounts only. */
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
            i++;
        }

        /* peak_block_cycles stays 0 on host: only the on-device DWT counter
         * (SPECTRAL_RESTRICTED_PROFILE) measures block cycles; the host never
         * fabricates them. Track the active peak only. */
        if (num_active > ops.peak_block_active) ops.peak_block_active = num_active;
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

        /* Embedded-target perf/memory estimate over the loaded segment set.
         * Recorded, not printed: console output belongs to the caller
         * (embedded_sim_last_report), never to a synthesis backend. */
        g_last_report.cfg = *cfg;
        g_last_report.est = embedded_perf_estimate(
            cfg, &ops, out_len, loaded, peak_active, elapsed);
        g_last_report.mem = embedded_memory_usage(
            loaded,
            block_size,
            SPECTRAL_OSC_LUT_BITS,
            SIMULATION_MAX_ACTIVE,
            cfg->max_memory_kb
        );
        g_last_report.valid = 1;
    }

    return SPECTRAL_OK;
}

int embedded_sim_last_report(EmbeddedTargetConfig* cfg,
                             EmbeddedPerfEstimate* est,
                             EmbeddedMemoryUsage* mem) {
    if (!g_last_report.valid || !cfg || !est || !mem) return 0;
    *cfg = g_last_report.cfg;
    *est = g_last_report.est;
    *mem = g_last_report.mem;
    return 1;
}

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
    return synth_cpu(sa, out_buffer, out_len, stretch, pitch,
                     timbre, n_threads, t_synth);
}
#endif
