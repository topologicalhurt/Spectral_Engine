/* spectral_hash_xx32_xx3.c
 *
 * Pattern: Lightweight strategy: capability Registry + Platform Adapter + Explicit Lifecycle.
 *
 * - Capability registry: method descriptors define which method IDs exist and
 *   whether they are available on this build target.
 * - Platform adapter: one thin internal state adapter maps the same lifecycle
 *   API onto XXH3 (host-file builds) or XXH32 (embedded/sim builds).
 * - Explicit lifecycle: init/reset -> update (0..N) -> digest -> destroy.
 *
 * On embedded/sim builds the XXH32 state is stored inline in SpectralHashFileMethod
 * (_embedded_state field) to avoid any heap allocation.
 *
 * This keeps public behavior stable while isolating platform differences and
 * making unsupported methods fail deterministically at lookup/init time.
 */
#include "spectral_hash_xx32_xx3.h"
#include <stdlib.h>  /* free for full_direct host buffer */
#include <stdint.h>
#include "spectral_utils.h"

/* ---------------------------------------------------------------------------
 * Internal state helpers
 * ---------------------------------------------------------------------------*/

static SpectralError spectral_hash_state_reset(void* state)
{
    if (!state) {
        return SPECTRAL_ERR_NOTINIT;
    }
#if SPECTRAL_HASH_HAS_HOST_FILE_API
    return (XXH3_64bits_reset((XXH3_state_t*)state) == XXH_OK)
        ? SPECTRAL_OK : SPECTRAL_ERR_PARAM;
#else
    return (XXH32_reset((XXH32_state_t*)state, 0u) == XXH_OK)
        ? SPECTRAL_OK : SPECTRAL_ERR_PARAM;
#endif
}

static SpectralError spectral_hash_state_update(void* state, const void* data, size_t len)
{
    if (!state) {
        return SPECTRAL_ERR_NOTINIT;
    }
#if SPECTRAL_HASH_HAS_HOST_FILE_API
    return (XXH3_64bits_update((XXH3_state_t*)state, data, len) == XXH_OK)
        ? SPECTRAL_OK : SPECTRAL_ERR_PARAM;
#else
    return (XXH32_update((XXH32_state_t*)state, data, len) == XXH_OK)
        ? SPECTRAL_OK : SPECTRAL_ERR_PARAM;
#endif
}

static SpectralError spectral_hash_state_digest(const void* state, SpectralHashDigest* out_digest)
{
    if (!state || !out_digest) {
        return SPECTRAL_ERR_PARAM;
    }
#if SPECTRAL_HASH_HAS_HOST_FILE_API
    *out_digest = XXH3_64bits_digest((const XXH3_state_t*)state);
#else
    *out_digest = (uint64_t)XXH32_digest((const XXH32_state_t*)state);
#endif
    return SPECTRAL_OK;
}

/* On embedded/sim builds state points into method->_embedded_state (inline storage),
 * so nothing is freed.  On host builds state is a heap-allocated XXH3_state_t. */
static void spectral_hash_state_free(void* state)
{
    if (!state) {
        return;
    }
#if SPECTRAL_HASH_HAS_HOST_FILE_API
    (void)XXH3_freeState((XXH3_state_t*)state);
#endif
}

/* ---------------------------------------------------------------------------
 * Core lifecycle impls
 * ---------------------------------------------------------------------------*/

static SpectralError spectral_hash_method_reset_impl(SpectralHashFileMethod* method)
{
    if (!method) {
        return SPECTRAL_ERR_PARAM;
    }

#if SPECTRAL_HASH_HAS_HOST_FILE_API
    if (!method->state) {
        method->state = (void*)XXH3_createState();
        if (!method->state) {
            return SPECTRAL_ERR_MEMORY;
        }
    }
#else
    /* On embedded/sim builds use the inline state — no heap allocation. */
    method->state = &method->_embedded_state;
#endif

    {
        SpectralError err = spectral_hash_state_reset(method->state);
        if (err != SPECTRAL_OK) {
            return err;
        }
    }

    method->initialized = 1;
    return SPECTRAL_OK;
}

static SpectralError spectral_hash_method_update_impl(
    SpectralHashFileMethod* method,
    const void* data,
    size_t len)
{
    if (!method) {
        return SPECTRAL_ERR_PARAM;
    }

    if (!method->initialized) {
        return SPECTRAL_ERR_NOTINIT;
    }

    if (len == 0u) {
        return SPECTRAL_OK;
    }

    if (!data) {
        return SPECTRAL_ERR_PARAM;
    }

    {
        SpectralError err = spectral_hash_state_update(method->state, data, len);
        if (err != SPECTRAL_OK) {
            return err;
        }
    }

    return SPECTRAL_OK;
}

static SpectralError spectral_hash_method_digest_impl(
    const SpectralHashFileMethod* method,
    SpectralHashDigest* out_digest)
{
    if (!method || !out_digest) {
        return SPECTRAL_ERR_PARAM;
    }

    if (!method->initialized) {
        return SPECTRAL_ERR_NOTINIT;
    }

    return spectral_hash_state_digest(method->state, out_digest);
}

/* ---------------------------------------------------------------------------
 * File consume impls
 * ---------------------------------------------------------------------------*/

static SpectralError spectral_hash_method_consume_file_stream_impl(
    SpectralHashFileMethod* method,
    FILE* file)
{
    unsigned char chunk[SPECTRAL_HASH_FILE_IO_CHUNK_SIZE];
    size_t nread;
    SpectralError reset_err;

    if (!method || !file) {
        return SPECTRAL_ERR_PARAM;
    }

    reset_err = spectral_hash_method_reset_impl(method);
    if (reset_err != SPECTRAL_OK) {
        return reset_err;
    }

    while ((nread = spectral_fs_read(chunk, 1, sizeof(chunk), file)) > 0u) {
        SpectralError update_err = spectral_hash_method_update_impl(method, chunk, nread);
        if (update_err != SPECTRAL_OK) {
            return update_err;
        }
    }

    if (ferror(file)) {
        return SPECTRAL_ERR_FILE_READ;
    }

    return SPECTRAL_OK;
}

static SpectralError spectral_hash_method_consume_file_full_direct_impl(
    SpectralHashFileMethod* method,
    FILE* file)
{
    uint64_t start_pos = 0;
    uint64_t end_pos = 0;
    uint64_t total_len_u64 = 0;
    size_t total_len = 0;
    unsigned char* data = NULL;
    SpectralError err;

    if (!method || !file) {
        return SPECTRAL_ERR_PARAM;
    }

    if (spectral_fs_tell(file, &start_pos, SPECTRAL_ERR_FILE_READ) != SPECTRAL_OK) {
        return spectral_hash_method_consume_file_stream_impl(method, file);
    }

    if (spectral_fs_seek(file, 0, SEEK_END, SPECTRAL_ERR_FILE_READ) != SPECTRAL_OK) {
        return spectral_hash_method_consume_file_stream_impl(method, file);
    }

    if (spectral_fs_tell(file, &end_pos, SPECTRAL_ERR_FILE_READ) != SPECTRAL_OK) {
        return SPECTRAL_ERR_FILE_READ;
    }
    if (end_pos < start_pos) {
        return SPECTRAL_ERR_FILE_READ;
    }

    /* Full-direct hashing reads the whole remaining file into one heap buffer.
     * The file-position API reports uint64_t, but the allocation and update
     * contract is size_t.  Never narrow silently: if the region cannot fit in a
     * size_t buffer, seek back and use the streaming implementation instead. */
    total_len_u64 = end_pos - start_pos;
    if (start_pos > (uint64_t)INT64_MAX) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    if (total_len_u64 > (uint64_t)SIZE_MAX) {
        if (spectral_fs_seek(file, (int64_t)start_pos, SEEK_SET, SPECTRAL_ERR_FILE_READ) != SPECTRAL_OK) {
            return SPECTRAL_ERR_FILE_READ;
        }
        return spectral_hash_method_consume_file_stream_impl(method, file);
    }

    if (spectral_fs_seek(file, (int64_t)start_pos, SEEK_SET, SPECTRAL_ERR_FILE_READ) != SPECTRAL_OK) {
        return SPECTRAL_ERR_FILE_READ;
    }

    total_len = (size_t)total_len_u64;
    if (total_len > 0u) {
        data = (unsigned char*)spectral_malloc_array(total_len, 1);
        if (!data) {
            return SPECTRAL_ERR_MEMORY;
        }
        if (spectral_fs_read(data, 1, total_len, file) != total_len) {
            free(data);
            return SPECTRAL_ERR_FILE_READ;
        }
    }

    /* reset then single-shot update; data may be NULL for empty files (update
     * short-circuits on len==0 and data==NULL so the NULL check is not needed). */
    err = spectral_hash_method_reset_impl(method);
    if (err == SPECTRAL_OK && total_len > 0u) {
        err = spectral_hash_method_update_impl(method, data, total_len);
    }

    if (data) {
        free(data);
    }
    return err;
}

/* TODO: implement mmap-backed hashing; currently falls back to SPECTRAL_ERR_BACKEND_UNAVAIL.
 * When implemented: mmap the file, call update() over the mapped region, munmap. */
static SpectralError spectral_hash_method_consume_file_mmap_impl(
    SpectralHashFileMethod* method,
    FILE* file)
{
    (void)method;
    (void)file;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

/* ---------------------------------------------------------------------------
 * Descriptor table
 * ---------------------------------------------------------------------------*/

static const SpectralHashFileMethodDescriptor k_hash_file_method_desc[SPECTRAL_HASH_FILE_METHOD_COUNT] = {
    [SPECTRAL_HASH_FILE_FULL_DIRECT] = {
        .type      = SPECTRAL_HASH_FILE_FULL_DIRECT,
        .name      = "full_direct",
        .available = 1
    },
    /* FULL_MMAP is registered for future support but is not advertised as available
     * until consume_file_mmap_impl actually maps and hashes the file. */
    [SPECTRAL_HASH_FILE_FULL_MMAP] = {
        .type      = SPECTRAL_HASH_FILE_FULL_MMAP,
        .name      = "full_mmap",
        .available = 0
    },
    /* STREAM is always available: callers feed data via update() directly.
     * consume_file() is not available on embedded (no file API), but the
     * reset/update/digest path works on all targets. */
    [SPECTRAL_HASH_FILE_STREAM] = {
        .type      = SPECTRAL_HASH_FILE_STREAM,
        .name      = "stream",
        .available = 1
    }
};

size_t spectral_hash_file_method_descriptor_count(void)
{
    return (size_t)SPECTRAL_HASH_FILE_METHOD_COUNT;
}

const SpectralHashFileMethodDescriptor* spectral_hash_file_method_descriptors(void)
{
    return k_hash_file_method_desc;
}

SpectralError spectral_hash_file_method_get_descriptor(
    SpectralHashFileMethodType type,
    const SpectralHashFileMethodDescriptor** out_desc)
{
    if (!out_desc) {
        return SPECTRAL_ERR_PARAM;
    }

    if ((int)type < 0 || type >= SPECTRAL_HASH_FILE_METHOD_COUNT) {
        *out_desc = NULL;
        return SPECTRAL_ERR_PARAM;
    }

    *out_desc = &k_hash_file_method_desc[(size_t)type];
    if (!(*out_desc)->available) {
        return SPECTRAL_ERR_BACKEND_UNAVAIL;
    }

    return SPECTRAL_OK;
}

/* ---------------------------------------------------------------------------
 * Public lifecycle
 * ---------------------------------------------------------------------------*/

SpectralError spectral_hash_file_method_init(
    SpectralHashFileMethod* method,
    SpectralHashFileMethodType type)
{
    SpectralError err;
    const SpectralHashFileMethodDescriptor* desc = NULL;

    if (!method) {
        return SPECTRAL_ERR_PARAM;
    }

    if (method->state || method->initialized) {
        return SPECTRAL_ERR_BUSY;
    }

    err = spectral_hash_file_method_get_descriptor(type, &desc);
    if (err != SPECTRAL_OK) {
        return err;
    }

    method->state       = NULL;
    method->initialized = 0;
    method->type        = desc->type;

    return spectral_hash_file_method_reset(method);
}

SpectralError spectral_hash_file_method_reset(SpectralHashFileMethod* method)
{
    SpectralError err = SPECTRAL_OK;
    const SpectralHashFileMethodDescriptor* desc = NULL;

    if (!method) {
        return SPECTRAL_ERR_PARAM;
    }

    /* Public reset is part of the explicit lifecycle.  A zero-initialized object
     * has type METHOD_COUNT and must not be implicitly initialized by reset().
     * init() sets method->type before its internal first reset, so validating the
     * descriptor here preserves init->reset behavior while rejecting reset before
     * init and unavailable methods. */
    err = spectral_hash_file_method_get_descriptor(method->type, &desc);
    if (err != SPECTRAL_OK) {
        return err;
    }

    return spectral_hash_method_reset_impl(method);
}


SpectralError spectral_hash_file_method_update(
    SpectralHashFileMethod* method,
    const void* data,
    size_t len)
{
    if (!method) {
        return SPECTRAL_ERR_PARAM;
    }

    return spectral_hash_method_update_impl(method, data, len);
}

SpectralError spectral_hash_file_method_digest(
    const SpectralHashFileMethod* method,
    SpectralHashDigest* out_digest)
{
    if (!method) {
        return SPECTRAL_ERR_PARAM;
    }

    return spectral_hash_method_digest_impl(method, out_digest);
}

SpectralError spectral_hash_file_method_consume_file(
    SpectralHashFileMethod* method,
    FILE* file)
{
    SpectralError err = SPECTRAL_OK;
    const SpectralHashFileMethodDescriptor* desc = NULL;

    if (!method || !file) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!method->initialized) {
        return SPECTRAL_ERR_NOTINIT;
    }

    err = spectral_hash_file_method_get_descriptor(method->type, &desc);
    if (err != SPECTRAL_OK) {
        return err;
    }

    switch (desc->type) {
        case SPECTRAL_HASH_FILE_FULL_DIRECT:
            return spectral_hash_method_consume_file_full_direct_impl(method, file);
        case SPECTRAL_HASH_FILE_FULL_MMAP:
            return spectral_hash_method_consume_file_mmap_impl(method, file);
        case SPECTRAL_HASH_FILE_STREAM:
            return spectral_hash_method_consume_file_stream_impl(method, file);
        default:
            return SPECTRAL_ERR_BACKEND_UNAVAIL;
    }
}


void spectral_hash_file_method_destroy(SpectralHashFileMethod* method)
{
    if (!method) {
        return;
    }

    if (method->state) {
        spectral_hash_state_free(method->state);
        method->state = NULL;
    }

    method->initialized = 0;
    method->type        = SPECTRAL_HASH_FILE_METHOD_COUNT;
}

SpectralHashDigest spectral_hash_oneshot(const void* data, size_t len)
{
    static const unsigned char empty = 0u;
    const void* src = data;

    if (!src) {
        if (len > 0u) {
            return (SpectralHashDigest)0;
        }
        src = &empty;
    }

#if SPECTRAL_HASH_HAS_HOST_FILE_API
    return XXH3_64bits(src, len);
#else
    return (uint64_t)XXH32(src, len, 0u);
#endif
}

