#!/usr/bin/env python3
"""Extract innermost loop bodies from GCC ARM assembly for llvm-mca.

Reads a .s file containing `# LLVM-MCA-BEGIN <name>` / `# LLVM-MCA-END`
marker comments (emitted by kernel_wrappers.c). Within each marked region,
finds innermost loops — a backward branch `b<cond> .Lx` to an already-seen
label — and writes each loop body as its own mca region named
`<kernel>/<label>`. The branch itself is included: it occupies an issue slot
and belongs in cycles-per-iteration.

Steady-state caveat (by construction, not assumption): llvm-mca models each
region as an infinite repetition of the body — correct for the unrolled
sustain/pair/fade loops this targets, which iterate over block samples.

Usage: extract_loops.py <in.s> <out.s>
"""
import re
import sys

BEGIN_RE = re.compile(r"#\s*LLVM-MCA-BEGIN\s+(\S+)")
END_RE = re.compile(r"#\s*LLVM-MCA-END")
LABEL_RE = re.compile(r"^(\.?L[\w.]+):")
# Conditional/unconditional branch to a local label; excludes bl/blx/bx (calls,
# returns) and cbz/cbnz (forward-only by ISA definition).
BRANCH_RE = re.compile(r"^\s+(b(?:ne|eq|cs|cc|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)?(?:\.[wn])?)\s+(\.?L[\w.]+)\b")
DIRECTIVE_RE = re.compile(r"^\s*\.")


def regions(lines):
    name, buf = None, []
    for ln in lines:
        m = BEGIN_RE.search(ln)
        if m:
            name, buf = m.group(1), []
            continue
        if END_RE.search(ln):
            if name is not None:
                yield name, buf
            name = None
            continue
        if name is not None:
            buf.append(ln)


def innermost_loops(body):
    label_at = {}
    spans = []
    for i, ln in enumerate(body):
        lm = LABEL_RE.match(ln)
        if lm:
            label_at[lm.group(1)] = i
            continue
        bm = BRANCH_RE.match(ln)
        if bm and bm.group(2) in label_at:
            spans.append((label_at[bm.group(2)], i, bm.group(2)))
    inner = []
    for s in spans:
        if not any(o is not s and s[0] <= o[0] and o[1] <= s[1] for o in spans):
            inner.append(s)
    return inner


def main(src, dst):
    lines = open(src).read().splitlines()
    out = []
    count = 0
    for name, body in regions(lines):
        for start, end, label in innermost_loops(body):
            insns = [ln for ln in body[start + 1 : end + 1]
                     if not DIRECTIVE_RE.match(ln) and not LABEL_RE.match(ln) and ln.strip()]
            if not insns:
                continue
            out.append(f"# LLVM-MCA-BEGIN {name}/{label}")
            out.extend(insns)
            out.append("# LLVM-MCA-END")
            count += 1
    open(dst, "w").write("\t.syntax unified\n\t.thumb\n" + "\n".join(out) + "\n")
    if count == 0:
        sys.exit("extract_loops: no loops found — marker or branch regex drift?")
    print(f"extract_loops: {count} innermost loop(s) -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
