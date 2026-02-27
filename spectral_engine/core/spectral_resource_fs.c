/* spectral_resource_fs.c - Resource identity and lookup layer. */
#include "spectral_resource_fs.h"
#include "spectral_utils.h"

#if SPECTRAL_HASH_HAS_HOST_FILE_API
#include <string.h>
#endif

/* ---------------------------------------------------------------------------
 * Locale-independent ASCII helpers used only by path canonicalization below.
 * Deliberately *not* using <ctype.h> tolower() to avoid locale-dependent
 * behaviour that would produce different canonical forms across machines.
 * ---------------------------------------------------------------------------*/
static inline int spectral_to_lower(int c)
{
    return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}

static inline char spectral_hex_nibble(unsigned n)
{
    n &= 0xfu;
    return (char)(n < 10u ? '0' + n : 'a' + (n - 10u));
}

/* ---------------------------------------------------------------------------
 * Path canonicalization — must match compress_path() in
 * tools/spectral_tools/generators/resource_hashes.py.
 *
 * Five transforms in order:
 *   (1) Lowercase
 *   (2) Strip control bytes \x00-\x1f and \x7f — makes \x01 safe as RLE sentinel
 *   (3) Normalize backslash to forward slash
 *   (4) Component-wise resolution:
 *         a. Strip trailing spaces BEFORE ".." check — catches the NTFS bypass
 *            where ".. " (dot-dot-space) names the parent directory on Windows.
 *         b. Resolve ".." by popping the last accepted component.
 *         c. Skip "." and empty components ("//", "./").
 *         d. Strip trailing dots — NTFS silently strips them ("file." == "file").
 *         e. Skip components that became empty after dot-stripping (e.g. "...").
 *   (5) Generalized RLE: runs of N >= 2 equal bytes encoded as a fixed 4-byte
 *       token: \x01 <byte> <hi_nibble_hex> <lo_nibble_hex> (lowercase hex).
 *       Runs > 255 are split into chained tokens; single chars pass through.
 *
 * Both sides must produce the identical byte sequence or embedded ID lookups
 * will silently fail.  Any change here must be reflected in compress_path().
 * ---------------------------------------------------------------------------*/

/* Worst-case RLE expansion is 2x (a run-of-2 encodes as 4 bytes).
 * A 512-byte normalized path produces at most a 1024-byte canonical form. */
#define SPECTRAL_CANONICAL_PATH_SIZE 1024u

/* Phase 1+2+3: lowercase, strip controls, normalize separators into scratch[]. */
static size_t spectral_path_p1(const char* path, char* scratch, size_t sz)
{
    const char* p;
    size_t w = 0u;
    for (p = path; *p && w + 1u < sz; p++) {
        unsigned char c = (unsigned char)*p;
        if (c <= 0x1fu || c == 0x7fu) { continue; }
        c = (unsigned char)spectral_to_lower((int)c);
        scratch[w++] = (c == (unsigned char)'\\') ? '/' : (char)c;
    }
    if (w < sz) { scratch[w] = '\0'; } else if (sz) { scratch[sz - 1u] = '\0'; }
    return w;
}

/* Phase 4: component-wise resolution into resolved[].
 * comp_starts[i] = resolved_len before component i's leading '/',
 * enabling O(1) pop for "..".
 * Cap: paths with more than 256 components are silently truncated at component
 * 257 onward.  For a 1024-byte scratch buffer the theoretical max is ~512 single-
 * char components, but real resource paths are never close to this limit. */
static size_t spectral_path_p2(
    const char* scratch, size_t slen, char* resolved, size_t rsz)
{
    size_t comp_starts[256];
    size_t comp_count = 0u;
    size_t rw = 0u;
    size_t i = 0u;

    while (i <= slen) {
        size_t j, end_sp, end_dot, k;

        j = i;
        while (j < slen && scratch[j] != '/') { j++; }

        /* (4a) strip trailing spaces — must precede ".." check */
        end_sp = j;
        while (end_sp > i && scratch[end_sp - 1u] == ' ') { end_sp--; }

        if (end_sp == i) {
            /* (4c) empty: skip */
        } else if (end_sp - i == 1u && scratch[i] == '.') {
            /* (4c) ".": skip */
        } else if (end_sp - i == 2u &&
                   scratch[i] == '.' && scratch[i + 1u] == '.') {
            /* (4b) "..": pop */
            if (comp_count > 0u) { comp_count--; rw = comp_starts[comp_count]; }
        } else {
            /* (4d) strip trailing dots */
            end_dot = end_sp;
            while (end_dot > i && scratch[end_dot - 1u] == '.') { end_dot--; }

            if (end_dot > i && comp_count < 256u) {
                /* (4f) valid: record position, append separator + chars */
                comp_starts[comp_count] = rw;
                if (comp_count > 0u && rw + 1u < rsz) { resolved[rw++] = '/'; }
                for (k = i; k < end_dot && rw + 1u < rsz; k++) {
                    resolved[rw++] = scratch[k];
                }
                comp_count++;
            }
            /* (4e) end_dot == i: all-dots component ("...") — skip */
        }
        i = j + 1u;
    }

    if (rw < rsz) { resolved[rw] = '\0'; } else if (rsz) { resolved[rsz - 1u] = '\0'; }
    return rw;
}

/* Phase 5: generalized RLE into out[].
 * Token: \x01 <char> <hi_nibble_hex> <lo_nibble_hex>  (fixed 4 bytes).
 * Single chars and chars near the buffer boundary pass through unchanged. */
static size_t spectral_path_p3(
    const char* resolved, size_t rlen, char* out, size_t out_size)
{
    size_t r = 0u;
    size_t w = 0u;

    while (r < rlen) {
        char c = resolved[r];
        size_t run = 0u;
        size_t remaining;
        while (r + run < rlen && resolved[r + run] == c) { run++; }
        r += run;

        remaining = run;
        while (remaining > 0u) {
            size_t chunk = (remaining > 255u) ? 255u : remaining;
            if (chunk >= 2u && w + 4u < out_size) {
                out[w++] = '\x01';
                out[w++] = c;
                out[w++] = spectral_hex_nibble((unsigned)chunk >> 4u);
                out[w++] = spectral_hex_nibble((unsigned)chunk & 0xfu);
                remaining -= chunk;
            } else if (chunk == 1u && w + 1u < out_size) {
                /* Only emit a literal for a genuine run-of-1.  Never fall back
                 * from an RLE token to literals: Python always encodes runs >= 2
                 * as tokens, so falling back would produce a different canonical
                 * form and therefore a different FNV-1a ID. */
                out[w++] = c;
                remaining--;
            } else {
                break; /* out of space: truncate rather than emit wrong encoding */
            }
        }
    }

    if (w < out_size) { out[w] = '\0'; } else if (out_size) { out[out_size - 1u] = '\0'; }
    return w;
}

size_t spectral_resource_path_canonical(
    const char* path, char* out, size_t out_size)
{
    char scratch[SPECTRAL_CANONICAL_PATH_SIZE];
    char resolved[SPECTRAL_CANONICAL_PATH_SIZE];
    size_t slen, rlen;

    if (!path || out_size == 0u) {
        if (out && out_size) { out[0] = '\0'; }
        return 0u;
    }

    slen = spectral_path_p1(path, scratch, sizeof(scratch));
    rlen = spectral_path_p2(scratch, slen, resolved, sizeof(resolved));
    return spectral_path_p3(resolved, rlen, out, out_size);
}

/* FNV-1a (32-bit) over the canonicalized path.
 *
 * The same FNV constants and the same canonicalization (compress_path) are
 * used by tools/spectral_tools/generators/resource_hashes.py so IDs are
 * identical across host and embedded
 * builds.  Returns 0 for a NULL path. */
SpectralResourceFileId spectral_resource_file_id_from_path(const char* path)
{
    char canonical[SPECTRAL_CANONICAL_PATH_SIZE];
    const unsigned char* p;
    uint32_t h = SPECTRAL_FNV1A_32_OFFSET_BASIS;

    if (!path) {
        return 0u;
    }

    spectral_resource_path_canonical(path, canonical, sizeof(canonical));
    p = (const unsigned char*)canonical;

    while (*p) {
        h ^= (uint32_t)(*p++);
        h *= SPECTRAL_FNV1A_32_PRIME;
    }

    return h;
}

#if SPECTRAL_HASH_HAS_HOST_FILE_API
const SpectralResourceFsEntry* spectral_resource_fs_find_by_path(const char* path)
{
    size_t i;
    if (!path) {
        return NULL;
    }
    for (i = 0; i < spectral_resource_hashes_count; i++) {
        if (strcmp(spectral_resource_hashes[i].path, path) == 0) {
            return &spectral_resource_hashes[i];
        }
    }
    return NULL;
}
#else
const SpectralResourceFsEntry* spectral_resource_fs_find_by_id(SpectralResourceFileId id)
{
    size_t i;
    for (i = 0; i < spectral_resource_hashes_count; i++) {
        if (spectral_resource_hashes[i].file_id == id) {
            return &spectral_resource_hashes[i];
        }
    }
    return NULL;
}
#endif
