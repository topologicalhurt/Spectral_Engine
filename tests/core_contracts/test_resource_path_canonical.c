/* test_resource_path_canonical.c - path canonicalization is a security boundary.
 *
 * spectral_resource_path_canonical() turns an untrusted resource-relative path
 * into the byte string the FNV-1a resource ID is computed over. Its component
 * resolution is the defense against path-traversal escapes (../, the NTFS
 * dot-dot-space and trailing-dot bypasses) and its control-byte strip keeps the
 * \x01 RLE sentinel unforgeable. None of that was exercised: the only coverage
 * was a 3-path round-trip whose inputs contain no "..", control bytes, trailing
 * dots, or runs, and the verify_resource_hashes target canonicalizes via this
 * same C code (C-against-C, so it cannot catch a transform regression).
 *
 * This pins the transform against an INDEPENDENT, hand-computed expectation:
 *   - exact canonical bytes for each phase (lowercase, separator, control strip,
 *     "." skip, ".." pop, trailing-dot strip, dot-dot-space, RLE token, >255 run
 *     chaining);
 *   - a structural security invariant over a traversal battery: a canonical form
 *     never contains ".." and never begins with a separator, so no input can
 *     name a parent of the resource root.
 *
 * The RLE token format (mirrored from spectral_resource_fs.c, NOT shared with it
 * so a drift in either is caught): runs of N>=2 equal bytes -> the 4-byte token
 * \x01 <byte> <hi_nibble_hex> <lo_nibble_hex> (lowercase hex); N>255 chains at 255.
 *
 * Run: cmake --build build --target resource_path_canonical_test
 *      && ctest --test-dir build -R resource_path_canonical
 */
#include "spectral_resource_fs.h"

#include <stdio.h>
#include <string.h>

#include "../support/check.h"

/* Canonicalize and compare against an expected byte sequence of known length
 * (expected may contain embedded \x01 RLE sentinels, so length is explicit). */
static void expect_canon(const char* in, const char* exp, size_t exp_len) {
    char out[SPECTRAL_CANONICAL_PATH_SIZE];
    size_t n = spectral_resource_path_canonical(in, out, sizeof(out));
    int ok = (n == exp_len) && (memcmp(out, exp, exp_len) == 0);
    if (!ok) {
        printf("  FAIL: canon(\"%s\") len=%zu (exp %zu) bytes=", in, n, exp_len);
        for (size_t i = 0; i < n; i++) {
            unsigned c = (unsigned char)out[i];
            if (c >= 0x20 && c < 0x7f) printf("%c", (char)c); else printf("\\x%02x", c);
        }
        printf("\n");
    }
    CHECK(ok, "canon(\"%s\") mismatch", in);
}

/* The load-bearing security property: a canonical form can never escape the
 * resource root. It must contain no ".." substring and must not begin with a
 * separator (which would name an absolute / parent path). */
static void expect_contained(const char* in) {
    char out[SPECTRAL_CANONICAL_PATH_SIZE];
    size_t n = spectral_resource_path_canonical(in, out, sizeof(out));
    out[(n < sizeof(out)) ? n : sizeof(out) - 1] = '\0';
    int has_dotdot = (strstr(out, "..") != NULL);
    int leads_sep  = (n > 0 && (out[0] == '/' || out[0] == '\\'));
    CHECK(!has_dotdot, "canon(\"%s\") = \"%s\" still contains \"..\" (traversal escape)", in, out);
    CHECK(!leads_sep,  "canon(\"%s\") = \"%s\" begins with a separator (escapes root)", in, out);
}

int main(void) {
    printf("== resource path canonicalization (security boundary) ==\n");

    /* --- exact-byte transform contract (inputs chosen with no adjacent dups so
     * the RLE pass is a passthrough and expected == resolved bytes) --- */
    expect_canon("ETC/X",        "etc/x",   5);   /* lowercase */
    expect_canon("a\\b\\c",      "a/b/c",   5);   /* backslash -> forward slash */
    expect_canon("a\x07x/c",     "ax/c",    4);   /* control byte stripped */
    expect_canon("./a/./x",      "a/x",     3);   /* "." components skipped */
    expect_canon("a//x",         "a/x",     3);   /* empty component skipped */
    expect_canon("fox./bar",     "fox/bar", 7);   /* trailing dot stripped (NTFS) */
    expect_canon("a/.../x",      "a/x",     3);   /* all-dots component -> empty -> skipped */

    /* ".." popping (traversal containment), exact form */
    expect_canon("a/b/../x",     "a/x",     3);   /* one pop */
    expect_canon("a/b/../../x",  "x",       1);   /* two pops to root */
    expect_canon("../../../x",   "x",       1);   /* ".." at root is absorbed, not an escape */
    expect_canon("a/.. /b",      "b",       1);   /* dot-dot-SPACE: space stripped, then pops */

    /* --- RLE token contract (independent of the impl's encoder) --- */
    expect_canon("aaa",   "\x01" "a03",       4);  /* run of 3 -> one token */
    expect_canon("fooo",  "f\x01" "o03",      5);  /* literal f + run-of-3 token */
    expect_canon("xx",    "\x01" "x02",       4);  /* minimum run (2) is a token, not literals */
    {
        /* run of 300 'a' -> 255-chunk token then 45-chunk token (chaining) */
        char big[301];
        memset(big, 'a', 300);
        big[300] = '\0';
        expect_canon(big, "\x01" "aff" "\x01" "a2d", 8);
    }

    /* --- security battery: none of these may yield a ".."-bearing or
     * separator-leading canonical form --- */
    static const char* const traversal[] = {
        "../../../etc/passwd",
        "..\\..\\windows\\system32",
        "a/../../../../../../etc/shadow",
        ".. /.. /secret",            /* dot-dot-space chain */
        "foo/...././bar",            /* mixed dot tricks */
        "/abs/escape",
        "....//....//x",
    };
    for (size_t i = 0; i < sizeof(traversal) / sizeof(traversal[0]); i++) {
        expect_contained(traversal[i]);
    }

    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
