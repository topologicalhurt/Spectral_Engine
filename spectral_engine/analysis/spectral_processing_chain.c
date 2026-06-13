/* spectral_processing_chain.c - Optional segment processing mask pipeline */
#include "spectral_processing_chain.h"
#include "spectral_proc_adaptive_track_density.h"
#include "spectral_utils.h"
#include "spectral_log.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    SpectralProcessMask bit;
    const char* name;
} ProcessMethod;

typedef SpectralError (*ProcessApplyFn)(SegmentArray*, int, const SpectralProcessParams*);

typedef struct {
    SpectralProcessMask bit;
    const char* name;
    ProcessApplyFn apply;
} ProcessStage;

static const ProcessMethod k_methods[] = {
    { SPECTRAL_PROC_SERRA_SMITH_1990,     "serra_smith_1990" },
    { SPECTRAL_PROC_JOHNSTON_1988,        "johnston_1988" },
    { SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY, "adaptive_track_density" },
    { SPECTRAL_PROC_REASSIGNED_TRACKING,  "reassigned" },
    { SPECTRAL_PROC_HYBRID_RENDER_SWITCH, "hybrid_render" },
    { SPECTRAL_PROC_EVENT_BUCKET_SCHED,   "event_bucket" },
    { SPECTRAL_PROC_HIGHER_ORDER_INTERP,  "higher_order_interp" },
    { SPECTRAL_PROC_QNOISE_SHAPING,       "qnoise_shaping" }
};

/* Stages with a REAL implementation. Every other named token parses (mask
 * values stay stable) but has no stage: requesting one fails loudly below —
 * a processing method that silently does nothing is worse than an error. */
static const ProcessStage k_stages[] = {
    { SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY,"adaptive_track_density",spectral_proc_adaptive_track_density_apply }
};

void spectral_process_params_default(SpectralProcessParams* params) {
    if (!params) return;
    params->deterministic_residual_db = -60.0f;
    params->psychoacoustic_margin_db = 3.0f;
    params->adaptive_max_segments = 0;
    params->adaptive_keep_ratio = 1.0f;
    params->hop = 0;
    params->n_fft = 0;
}

static int token_equal_ci(const char* a, const char* b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++;
        b++;
    }
    return (*a == '\0' && *b == '\0');
}

static int parse_named_token(const char* token, SpectralProcessMask* out_bit) {
    if (token_equal_ci(token, "none")) {
        *out_bit = SPECTRAL_PROC_NONE;
        return 1;
    }
    if (token_equal_ci(token, "default")) {
        *out_bit = SPECTRAL_PROC_NONE;
        return 1;
    }
    if (token_equal_ci(token, "all")) {
        *out_bit = SPECTRAL_PROC_ALL_KNOWN;
        return 1;
    }
    if (token_equal_ci(token, "det") || token_equal_ci(token, "deterministic_model") ||
        token_equal_ci(token, "deterministic") || token_equal_ci(token, "sms") ||
        token_equal_ci(token, "serra_smith")) {
        *out_bit = SPECTRAL_PROC_SERRA_SMITH_1990;
        return 1;
    }
    if (token_equal_ci(token, "masking") || token_equal_ci(token, "psycho") ||
        token_equal_ci(token, "psychoacoustic")) {
        *out_bit = SPECTRAL_PROC_JOHNSTON_1988;
        return 1;
    }
    if (token_equal_ci(token, "adaptive") || token_equal_ci(token, "adaptive_density") ||
        token_equal_ci(token, "adaptive_track_density") ||
        token_equal_ci(token, "mcaulay") || token_equal_ci(token, "quatieri") ||
        token_equal_ci(token, "mcaulay_quatieri_1986")) {
        *out_bit = SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY;
        return 1;
    }
    if (token_equal_ci(token, "hybrid")) {
        *out_bit = SPECTRAL_PROC_HYBRID_RENDER_SWITCH;
        return 1;
    }
    if (token_equal_ci(token, "scheduler") || token_equal_ci(token, "events")) {
        *out_bit = SPECTRAL_PROC_EVENT_BUCKET_SCHED;
        return 1;
    }
    if (token_equal_ci(token, "interp")) {
        *out_bit = SPECTRAL_PROC_HIGHER_ORDER_INTERP;
        return 1;
    }
    if (token_equal_ci(token, "noise_shaping") || token_equal_ci(token, "qnoise")) {
        *out_bit = SPECTRAL_PROC_QNOISE_SHAPING;
        return 1;
    }

    for (size_t i = 0; i < sizeof(k_methods) / sizeof(k_methods[0]); i++) {
        if (token_equal_ci(token, k_methods[i].name)) {
            *out_bit = k_methods[i].bit;
            return 1;
        }
    }
    return 0;
}

int spectral_process_mask_parse(const char* spec, SpectralProcessMask* out_mask) {
    if (!spec || !out_mask) return 0;

    if ((spec[0] >= '0' && spec[0] <= '9')) {
        char* end = NULL;
        unsigned long v = strtoul(spec, &end, 0);
        if (!end || *end != '\0') return 0;
        if ((v & ~(unsigned long)SPECTRAL_PROC_ALL_KNOWN) != 0UL) return 0;
        *out_mask = (SpectralProcessMask)v;
        return 1;
    }

    size_t n = strlen(spec);
    size_t buf_len = 0;
    if (!spectral_size_add(n, 1u, &buf_len)) return 0;
    char* buf = (char*)spectral_malloc_array(buf_len, sizeof(char));
    if (!buf) return 0;
    memcpy(buf, spec, buf_len);

    SpectralProcessMask mask = 0;
    int saw_none = 0;
    int ok = 1;
    char* save = NULL;
    for (char* tok = strtok_r(buf, ",+| ", &save); tok; tok = strtok_r(NULL, ",+| ", &save)) {
        SpectralProcessMask bit = 0;
        if (!parse_named_token(tok, &bit)) {
            ok = 0;
            break;
        }
        if (bit == SPECTRAL_PROC_NONE) {
            if (mask != 0) {
                ok = 0;
                break;
            }
            saw_none = 1;
            continue;
        }
        if (saw_none) {
            ok = 0;
            break;
        }
        mask |= bit;
    }

    free(buf);
    if (!ok) return 0;
    *out_mask = mask;
    return 1;
}

void spectral_process_mask_to_string(SpectralProcessMask mask, char* out, size_t out_size) {
    if (!out || out_size == 0) return;
    out[0] = '\0';

    if (mask == SPECTRAL_PROC_NONE) {
        snprintf(out, out_size, "none");
        return;
    }

    size_t used = 0;
    for (size_t i = 0; i < sizeof(k_methods) / sizeof(k_methods[0]); i++) {
        if (!(mask & k_methods[i].bit)) continue;
        const char* sep = (used == 0) ? "" : ",";
        int w = snprintf(out + used, (used < out_size) ? (out_size - used) : 0,
                         "%s%s", sep, k_methods[i].name);
        if (w < 0) break;
        used += (size_t)w;
        if (used >= out_size) break;
    }

    if (used == 0) {
        snprintf(out, out_size, "0x%08x", (unsigned)mask);
    }
}

SpectralError spectral_process_chain_apply(
    SegmentArray* sa,
    int sample_rate,
    int hop,
    int n_fft,
    SpectralProcessMask mask,
    SpectralProcessReport* report)
{
    if (!sa) return SPECTRAL_ERR_PARAM;
    if (mask & ~SPECTRAL_PROC_ALL_KNOWN) return SPECTRAL_ERR_PARAM;

    SpectralProcessParams params;
    spectral_process_params_default(&params);
    params.hop = hop;
    params.n_fft = n_fft;

    SpectralProcessReport local = {0};
    local.requested = mask;

    for (size_t i = 0; i < sizeof(k_stages) / sizeof(k_stages[0]); i++) {
        if (!(mask & k_stages[i].bit)) continue;
        SpectralError err = k_stages[i].apply(sa, sample_rate, &params);
        if (err != SPECTRAL_OK) return err;
        local.applied |= k_stages[i].bit;
    }

    local.pending = mask & ~local.applied;

    if (report) *report = local;
    if (local.pending != SPECTRAL_PROC_NONE) {
        char pending_buf[192] = {0};
        spectral_process_mask_to_string(local.pending, pending_buf,
                                        sizeof(pending_buf));
        SPECTRAL_LOG_ERROR_STDERR(
            "Processing stage(s) not implemented: %s", pending_buf);
        return SPECTRAL_ERR_PARAM;
    }
    return SPECTRAL_OK;
}
