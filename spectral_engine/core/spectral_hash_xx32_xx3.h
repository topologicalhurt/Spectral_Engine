/* spectral_hash_xx32_xx3.h - Streaming file-hash API.
 *
 * Wraps XXH3-64 (host builds) or XXH32 (embedded/sim builds) behind a
 * platform-adaptive lifecycle API:
 *
 *   SpectralHashFileMethod m = SPECTRAL_HASH_FILE_METHOD_ZERO_INIT;
 *   spectral_hash_file_method_init(&m, SPECTRAL_HASH_FILE_FULL_DIRECT);
 *   // m is ready for update() calls immediately after init().
 *   spectral_hash_file_method_update(&m, data, len);
 *   spectral_hash_file_method_digest(&m, &digest);
 *   spectral_hash_file_method_destroy(&m);
 *
 * SpectralHashDigest is 8 bytes on host (XXH64_hash_t) and 4 bytes on
 * embedded/sim (XXH32_hash_t).  It matches SpectralHashStoredDigest in
 * spectral_resource_fs.h — they are kept as distinct types because one is
 * a runtime compute result and the other is a table storage type.
 *
 * Include path: third_party/xxHash must be in the compiler's include
 * search path (added via SPECTRAL_XXHASH_INCLUDE_DIR in source-manifest.cmake).
 */
#ifndef SPECTRAL_HASH_XX32_XX3_H
#define SPECTRAL_HASH_XX32_XX3_H

#include <stddef.h>
#include "spectral_config.h"
#include "spectral_error.h"

/* XXH_STATIC_LINKING_ONLY exposes the complete struct definitions for
 * XXH32_state_s and XXH3_state_s.  We need this unconditionally because
 * on embedded builds the XXH32 state is stored inline in SpectralHashFileMethod
 * (to avoid heap allocation) and the compiler requires a complete type for
 * struct member sizing.  On host builds the define is harmless — the extra
 * declarations are simply unused. */
#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"

#if SPECTRAL_HASH_HAS_HOST_FILE_API
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpectralHashFileMethod SpectralHashFileMethod;
typedef struct SpectralHashFileMethodDescriptor SpectralHashFileMethodDescriptor;

#if SPECTRAL_HASH_HAS_HOST_FILE_API
typedef XXH64_hash_t SpectralHashDigest;
#else
typedef XXH32_hash_t SpectralHashDigest;
#endif

typedef enum SpectralHashFileMethodType {
    SPECTRAL_HASH_FILE_FULL_DIRECT = 0,
    SPECTRAL_HASH_FILE_FULL_MMAP   = 1,  /* available=0 when !SPECTRAL_HASH_HAS_MMAP */
    SPECTRAL_HASH_FILE_STREAM      = 2,
    SPECTRAL_HASH_FILE_METHOD_COUNT = 3
} SpectralHashFileMethodType;

struct SpectralHashFileMethod {
    void*                      state;
    int                        initialized;
    SpectralHashFileMethodType type;
#if !SPECTRAL_HASH_HAS_HOST_FILE_API
    /* On embedded/sim builds the XXH32 state lives here to avoid malloc.
     * reset_impl() points method->state at this field instead of allocating.
     * destroy() sets state=NULL; a subsequent reset() re-points it here. */
    XXH32_state_t              _embedded_state;
#endif
};

/* Zero-initialize with designated initializers so _embedded_state (and any
 * future fields) are implicitly zeroed without being listed explicitly. */
#define SPECTRAL_HASH_FILE_METHOD_ZERO_INIT \
    { .state = NULL, .initialized = 0, .type = SPECTRAL_HASH_FILE_METHOD_COUNT }

struct SpectralHashFileMethodDescriptor {
    SpectralHashFileMethodType type;
    const char*                name;
    /* 1 if usable on this build target; 0 if the method is registered but
     * unavailable (e.g. mmap on non-POSIX, full_direct on embedded).
     *
     * Note for SPECTRAL_HASH_FILE_STREAM: available=1 on all targets, meaning
     * the reset/update/digest lifecycle API is fully functional.  However,
     * consume_file() is only declared when SPECTRAL_HASH_HAS_HOST_FILE_API is
     * set; on embedded targets there is no file I/O layer and consume_file()
     * does not exist.  Callers on embedded feed data through update() directly. */
    int                        available;
};

size_t spectral_hash_file_method_descriptor_count(void);
const SpectralHashFileMethodDescriptor* spectral_hash_file_method_descriptors(void);
SpectralError spectral_hash_file_method_get_descriptor(
    SpectralHashFileMethodType type,
    const SpectralHashFileMethodDescriptor** out_desc);

/* Initialize a zero-initialized method object for the given method type.
 * Callers must create objects with SPECTRAL_HASH_FILE_METHOD_ZERO_INIT.
 * init() also performs the first reset, so the method is ready for update()
 * calls immediately.  Re-initializing an active object returns SPECTRAL_ERR_BUSY;
 * call destroy() first. */
SpectralError spectral_hash_file_method_init(
    SpectralHashFileMethod* method,
    SpectralHashFileMethodType type);

/* reset() clears accumulated state; the method can then accept new update() calls.
 * Calling reset() on an already-reset or freshly-init'd method is safe. */
SpectralError spectral_hash_file_method_reset(SpectralHashFileMethod* method);
SpectralError spectral_hash_file_method_update(
    SpectralHashFileMethod* method,
    const void* data,
    size_t len);
SpectralError spectral_hash_file_method_digest(
    const SpectralHashFileMethod* method,
    SpectralHashDigest* out_digest);
#if SPECTRAL_HASH_HAS_HOST_FILE_API
/* Hashes the entire file in one call using the method's preferred I/O strategy.
 * The method is reset before reading; digest can be retrieved afterward. */
SpectralError spectral_hash_file_method_consume_file(
    SpectralHashFileMethod* method,
    FILE* file);
#endif
void spectral_hash_file_method_destroy(SpectralHashFileMethod* method);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_HASH_XX32_XX3_H */
