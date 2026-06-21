/* QEMU TCG plugin: exact dynamic instruction + memory counts, attributed to
 * ELF symbol ranges (M7_PERF_MODEL_PLAN P2, Layer 1).
 *
 * Counts are MEASURED quantities: retired instructions, load/store bytes,
 * and unique 32B-line working sets of the real ARM binary. TCG has no
 * timing model — nothing here is a cycle. The working-set line counts feed
 * the P4 analytical cache layer (M7 D-cache lines are 32B); they are
 * footprints, not miss counts.
 *
 * Args:  range=name:0xstart:0xend   (repeatable; from arm-none-eabi-nm)
 *        out=path                   (report file; default stderr)
 *        lineshift=N                (REQUIRED: log2(cache line bytes); the
 *                                    driver derives it from the C SSOT
 *                                    daisy_seed_sdram.h — no copy lives here)
 *        trace=path                 (optional: full data-access address trace,
 *                                    one "L|S vaddr size" line per access —
 *                                    large; for offline cache simulation)
 * Data-address buckets (code/SSRAM23/0x60000000-bulk) come for free and feed
 * the P4 placement model. Single-vCPU machine: plain increments are exact. */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define MAX_RANGES 64
#define N_REGIONS 4

/* log2(cache line bytes); set by the REQUIRED lineshift= plugin arg from the
 * C SSOT (daisy_seed_sdram.h) — install fails without it (no silent copy). */
static unsigned line_shift = 0;

typedef struct {
    char name[64];
    uint64_t start, end;            /* [start, end) */
    uint64_t insns;
    uint64_t loads, stores;
    uint64_t load_bytes, store_bytes;
    GHashTable* lines;              /* unique 32B data lines touched (ld+st) */
} Range;

static Range ranges[MAX_RANGES];
static int n_ranges = 1;            /* ranges[0] = "<other>" catchall */
static char out_path[1024];
static FILE* trace_file = NULL;

/* Data-address region buckets (mps2-an500 map, verified via info mtree). */
static uint64_t region_bytes[N_REGIONS];   /* 0:code(0x0) 1:ssram23(0x2) 2:bulk(0x6) 3:other */
static GHashTable* region_lines[N_REGIONS];

static int range_of(uint64_t a) {
    for (int i = 1; i < n_ranges; i++)
        if (a >= ranges[i].start && a < ranges[i].end) return i;
    return 0;
}

static int region_of(uint64_t a) {
    if (a < 0x20000000ull) return 0;
    if (a < 0x30000000ull) return 1;
    if (a >= 0x60000000ull && a < 0x70000000ull) return 2;
    return 3;
}

static void exec_cb(unsigned int vcpu_index, void* udata) {
    (void)vcpu_index;
    ((Range*)udata)->insns++;
}

static void mem_cb(unsigned int vcpu_index, qemu_plugin_meminfo_t info,
                   uint64_t vaddr, void* udata) {
    (void)vcpu_index;
    Range* r = (Range*)udata;
    uint64_t bytes = 1ull << qemu_plugin_mem_size_shift(info);
    int is_store = qemu_plugin_mem_is_store(info);
    if (is_store) {
        r->stores++;
        r->store_bytes += bytes;
    } else {
        r->loads++;
        r->load_bytes += bytes;
    }
    /* An access can straddle a line boundary; record every line it touches. */
    uint64_t first_line = vaddr >> line_shift;
    uint64_t last_line = (vaddr + bytes - 1) >> line_shift;
    int region = region_of(vaddr);
    region_bytes[region] += bytes;
    for (uint64_t line = first_line; line <= last_line; line++) {
        g_hash_table_add(r->lines, GSIZE_TO_POINTER((gsize)line));
        g_hash_table_add(region_lines[region], GSIZE_TO_POINTER((gsize)line));
    }
    if (trace_file) {
        fprintf(trace_file, "%c %" PRIx64 " %" PRIu64 "\n",
                is_store ? 'S' : 'L', vaddr, bytes);
    }
}

static void tb_trans_cb(qemu_plugin_id_t id, struct qemu_plugin_tb* tb) {
    (void)id;
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn* insn = qemu_plugin_tb_get_insn(tb, i);
        Range* r = &ranges[range_of(qemu_plugin_insn_vaddr(insn))];
        qemu_plugin_register_vcpu_insn_exec_cb(insn, exec_cb,
                                               QEMU_PLUGIN_CB_NO_REGS, r);
        qemu_plugin_register_vcpu_mem_cb(insn, mem_cb, QEMU_PLUGIN_CB_NO_REGS,
                                         QEMU_PLUGIN_MEM_RW, r);
    }
}

static void atexit_cb(qemu_plugin_id_t id, void* p) {
    (void)id; (void)p;
    FILE* f = out_path[0] ? fopen(out_path, "w") : stderr;
    if (!f) f = stderr;
    fprintf(f, "# [measured: qemu-tcg dynamic counts] — not cycles\n");
    fprintf(f, "%-28s %14s %12s %12s %14s %14s %12s\n",
            "range", "insns", "loads", "stores", "load_bytes", "store_bytes",
            "lines32B");
    for (int i = 0; i < n_ranges; i++) {
        Range* r = &ranges[i];
        fprintf(f, "%-28s %14" PRIu64 " %12" PRIu64 " %12" PRIu64
                   " %14" PRIu64 " %14" PRIu64 " %12u\n",
                r->name, r->insns, r->loads, r->stores,
                r->load_bytes, r->store_bytes,
                g_hash_table_size(r->lines));
    }
    fprintf(f, "# data bytes by region: code=%" PRIu64 " ssram23=%" PRIu64
               " bulk60=%" PRIu64 " other=%" PRIu64 "\n",
            region_bytes[0], region_bytes[1], region_bytes[2], region_bytes[3]);
    fprintf(f, "# unique 32B lines by region: code=%u ssram23=%u bulk60=%u other=%u\n",
            g_hash_table_size(region_lines[0]), g_hash_table_size(region_lines[1]),
            g_hash_table_size(region_lines[2]), g_hash_table_size(region_lines[3]));
    if (f != stderr) fclose(f);
    if (trace_file) fclose(trace_file);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t* info,
                                           int argc, char** argv) {
    (void)info;
    snprintf(ranges[0].name, sizeof ranges[0].name, "<other>");
    for (int i = 0; i < MAX_RANGES; i++)
        ranges[i].lines = g_hash_table_new(g_direct_hash, g_direct_equal);
    for (int i = 0; i < N_REGIONS; i++)
        region_lines[i] = g_hash_table_new(g_direct_hash, g_direct_equal);

    for (int i = 0; i < argc; i++) {
        if (g_str_has_prefix(argv[i], "range=")) {
            if (n_ranges >= MAX_RANGES) {
                fprintf(stderr, "spectral_counts: too many ranges\n");
                return -1;
            }
            char* spec = argv[i] + 6;
            char** parts = g_strsplit(spec, ":", 3);
            if (!parts[0] || !parts[1] || !parts[2]) {
                fprintf(stderr, "spectral_counts: bad range '%s'\n", spec);
                g_strfreev(parts);
                return -1;
            }
            Range* r = &ranges[n_ranges++];
            snprintf(r->name, sizeof r->name, "%s", parts[0]);
            r->start = g_ascii_strtoull(parts[1], NULL, 16);
            r->end   = g_ascii_strtoull(parts[2], NULL, 16);
            g_strfreev(parts);
        } else if (g_str_has_prefix(argv[i], "out=")) {
            snprintf(out_path, sizeof out_path, "%s", argv[i] + 4);
        } else if (g_str_has_prefix(argv[i], "lineshift=")) {
            line_shift = (unsigned)g_ascii_strtoull(argv[i] + 10, NULL, 10);
        } else if (g_str_has_prefix(argv[i], "trace=")) {
            trace_file = fopen(argv[i] + 6, "w");
            if (!trace_file) {
                fprintf(stderr, "spectral_counts: cannot open trace file '%s'\n", argv[i] + 6);
                return -1;
            }
        }
    }
    if (line_shift == 0u || line_shift > 12u) {
        fprintf(stderr, "spectral_counts: lineshift=N (1..12) is required — "
                        "derive it from daisy_seed_sdram.h, do not hardcode\n");
        return -1;
    }
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans_cb);
    qemu_plugin_register_atexit_cb(id, atexit_cb, NULL);
    return 0;
}
