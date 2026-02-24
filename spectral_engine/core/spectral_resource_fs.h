/* spectral_resource_fs.h - Resource identity and lookup layer.
 *
 * This is the stable public interface between the rest of the engine and the
 * resource subsystem.  It abstracts the host/embedded identity split:
 *
 *   Host:     resources are identified by relative path string.
 *   Embedded: resources are identified by a 32-bit FNV-1a hash of that path.
 *             (no filesystem; the hash table is compiled into flash.)
 *
 * NOTE: FNV-1a is STRICTLY used for path-to-ID generation on Embedded.
 * For hashing the actual payload content, Host uses XXH3-64 and 
 * Embedded uses XXH32.
 *
 * The generated file (spectral_hash_resources_xx32_xx3.c) defines the actual
 * hash table.  All callers look up entries through this header's API rather
 * than touching the table directly, so the table format can evolve without
 * breaking call sites.
 */
#ifndef SPECTRAL_RESOURCE_FS_H
#define SPECTRAL_RESOURCE_FS_H

#include <stddef.h>
#include <stdint.h>
#include "spectral_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resource file identifier.
 * On host builds the human-readable path is available; on embedded builds
 * only this 32-bit ID (derived at build time from the path via FNV-1a) is
 * stored, since there is no filesystem to resolve string paths against. */
typedef uint32_t SpectralResourceFileId;

/* Width of the hash digest stored in the resource table.
 * Host builds use XXH3-64 (8 bytes); embedded/sim builds use XXH32 (4 bytes).
 * SpectralHashDigest in spectral_hash_xx32_xx3.h matches this width —
 * both exist because the resource table entry and the runtime compute result
 * are semantically distinct even though they are the same width. */
typedef uint64_t SpectralHashStoredDigest;

/* One row in the resource hash table.
 * The identity field is conditional on build target (see above). */
typedef struct SpectralResourceFsEntry {
#if SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM
    SpectralResourceFileId   file_id;
#else
    const char*              path;
#endif
    SpectralHashStoredDigest hash;
    size_t                   size;
} SpectralResourceFsEntry;

/* The resource hash table and its count.  Defined by the generated source;
 * treat as read-only — entries are resolved at build time. */
extern const SpectralResourceFsEntry spectral_resource_hashes[];
extern const size_t                  spectral_resource_hashes_count;

/* FNV-1a 32-bit algorithm constants used by spectral_resource_file_id_from_path().
 * MUST match the constants used in gen_resource_hashes.py::file_id_from_path()
 * — any divergence causes silent ID mismatches on embedded targets. */
#define SPECTRAL_FNV1A_32_OFFSET_BASIS  UINT32_C(2166136261)
#define SPECTRAL_FNV1A_32_PRIME         UINT32_C(16777619)

/* Compute the FNV-1a file ID for an arbitrary path string.
 * The path is canonicalized before hashing (lowercase, control-strip, path
 * normalization, generalized RLE) — see compress_path() in gen_resource_hashes.py
 * for the byte-identical transform.  Returns 0 for NULL. */
SpectralResourceFileId spectral_resource_file_id_from_path(const char* path);

/* Look up a resource entry by path (host) or pre-computed ID (embedded).
 * Returns NULL if not found.
 *
 * spectral_resource_fs_find_by_path(): performs an exact strcmp against the
 * stored path strings — which are the raw POSIX-relative paths as emitted by
 * gen_resource_hashes.py (e.g. "sounds/click.wav").  The path argument must
 * match byte-for-byte; no canonicalization is applied at lookup time.  If you
 * only have a user-supplied path that may differ in case or separators, derive
 * the file ID first with spectral_resource_file_id_from_path() and use the
 * embedded-style lookup instead. */
#if SPECTRAL_HASH_HAS_HOST_FILE_API
const SpectralResourceFsEntry* spectral_resource_fs_find_by_path(const char* path);
#else
const SpectralResourceFsEntry* spectral_resource_fs_find_by_id(SpectralResourceFileId id);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_RESOURCE_FS_H */
