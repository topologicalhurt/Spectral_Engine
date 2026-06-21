/* QEMU TCG plugin: per-PC dynamic instruction histogram.
 *
 * Counts retired instructions per program-counter address of the real ARM
 * binary under qemu-system-arm. The driver maps the hot PCs back to inlined
 * function + source line via addr2line on a -g ELF, giving "hottest functions
 * and lines" (DETERMINISM_SURFACE_PLAN P4). Counts are MEASURED (qemu-tcg);
 * nothing here is a cycle. Single-vCPU machine: plain increments are exact.
 *
 * Args: out=path   (histogram file: "PC COUNT" hex lines; default stderr)
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static GHashTable* pc_counts;     /* vaddr -> retired-instruction count */
static char out_path[1024];

static void exec_cb(unsigned int vcpu_index, void* udata) {
    (void)vcpu_index;
    gsize pc = GPOINTER_TO_SIZE(udata);
    gpointer cur = g_hash_table_lookup(pc_counts, GSIZE_TO_POINTER(pc));
    g_hash_table_insert(pc_counts, GSIZE_TO_POINTER(pc),
                        GSIZE_TO_POINTER(GPOINTER_TO_SIZE(cur) + 1));
}

static void tb_trans_cb(qemu_plugin_id_t id, struct qemu_plugin_tb* tb) {
    (void)id;
    size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn* insn = qemu_plugin_tb_get_insn(tb, i);
        uint64_t pc = qemu_plugin_insn_vaddr(insn);
        qemu_plugin_register_vcpu_insn_exec_cb(insn, exec_cb,
                                               QEMU_PLUGIN_CB_NO_REGS,
                                               GSIZE_TO_POINTER((gsize)pc));
    }
}

static void atexit_cb(qemu_plugin_id_t id, void* p) {
    (void)id; (void)p;
    FILE* f = out_path[0] ? fopen(out_path, "w") : stderr;
    if (!f) f = stderr;
    fprintf(f, "# [measured: qemu-tcg per-PC retired instructions] — not cycles\n");
    GHashTableIter it;
    gpointer key, val;
    g_hash_table_iter_init(&it, pc_counts);
    while (g_hash_table_iter_next(&it, &key, &val)) {
        fprintf(f, "%" PRIx64 " %" PRIu64 "\n",
                (uint64_t)GPOINTER_TO_SIZE(key), (uint64_t)GPOINTER_TO_SIZE(val));
    }
    if (f != stderr) fclose(f);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t* info,
                                           int argc, char** argv) {
    (void)info;
    pc_counts = g_hash_table_new(g_direct_hash, g_direct_equal);
    for (int i = 0; i < argc; i++) {
        if (g_str_has_prefix(argv[i], "out="))
            snprintf(out_path, sizeof out_path, "%s", argv[i] + 4);
    }
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans_cb);
    qemu_plugin_register_atexit_cb(id, atexit_cb, NULL);
    return 0;
}
