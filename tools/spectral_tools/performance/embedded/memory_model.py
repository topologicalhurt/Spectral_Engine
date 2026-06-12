"""Layer 3: analytical memory-stall model over Layer-1 address traces (P4).

Replays the QEMU plugin's data-access trace (program order, exact addresses)
through the Daisy placement and a Cortex-M7 D-cache model, pricing misses with
the CHOSEN hardware configuration. Nothing here is a measurement of hardware:
the trace is measured, the stalls are MODELED, and the report says which is
which (M7_PERF_MODEL_PLAN fidelity contract).

Constants are CENTRALIZED and parsed at runtime (C-truth rule, ADR-0002):
  - api/daisy_seed/daisy_seed_sdram.h — the ONE place for clock tree, FMC
    SDRAM timing (the maintainer-chosen best-performance set the firmware
    actually programs), SDRAM geometry, and cache geometry; per-value
    provenance lives next to each value in that header. Python holds no copy.
  - native/qemu/mps2_an500.ld — the block-marker address (MARKER region).
The only Python-side constants are model-side: the uncalibrated AXIM->FMC
bridge assumption, and libDaisy's STOCK timing set kept for the comparison row
(it is libDaisy's fact, not this repo's configuration).

Placement mirror (the counts rig's ldscript ⇄ Daisy):
  DATA  0x20000000  -> DTCM-class   (Daisy .dtcmram_bss: ctx, q63 accum, blk)
  BULK  0x60000000  -> SDRAM-class  (Daisy .sdram_bss: the segment store),
                       reached through the 4-way WBWA D-cache
  CODE  <0x20000000 -> flash-class  (rodata via D-cache; code via I-cache)

Stall accounting is reported as a RANGE, not a point: the M7 is non-blocking
(two data linefill buffers, merging store buffer [TRM]), so the true cost of N
misses lies between the bandwidth bound (transfers fully overlapped) and the
serial bound (every miss latency exposed). A point estimate would be theater.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ...core.utils import find_repo_root

SDRAM_HEADER_RELPATH = "api/daisy_seed/daisy_seed_sdram.h"
LDSCRIPT_RELPATH = (
    "tools/spectral_tools/performance/embedded/native/qemu/mps2_an500.ld"
)

# The header keys the model requires; a missing one means the SSOT contract
# moved and the model must fail loudly, not fall back to a stale copy.
_REQUIRED_DEFINES = (
    "SPECTRAL_DAISY_CPU_HZ", "SPECTRAL_DAISY_AXI_HZ", "SPECTRAL_DAISY_SDCLK_HZ",
    "SPECTRAL_DAISY_SDRAM_BUS_BITS", "SPECTRAL_DAISY_SDRAM_BANKS",
    "SPECTRAL_DAISY_SDRAM_ROW_BYTES",
    "SPECTRAL_DAISY_SDRAM_TMRD", "SPECTRAL_DAISY_SDRAM_TXSR",
    "SPECTRAL_DAISY_SDRAM_TRAS", "SPECTRAL_DAISY_SDRAM_TRC",
    "SPECTRAL_DAISY_SDRAM_TWR", "SPECTRAL_DAISY_SDRAM_TRP",
    "SPECTRAL_DAISY_SDRAM_TRCD", "SPECTRAL_DAISY_SDRAM_CAS",
    "SPECTRAL_DAISY_DCACHE_BYTES", "SPECTRAL_DAISY_DCACHE_WAYS",
    "SPECTRAL_DAISY_CACHE_LINE",
)

_DEFINE_RE = re.compile(r"^#define\s+(SPECTRAL_DAISY_\w+)\s+(\d+)u?\b", re.MULTILINE)
_MARKER_RE = re.compile(r"MARKER\s*\(\w+\)\s*:\s*ORIGIN\s*=\s*(0x[0-9A-Fa-f]+)")

# Model-side assumption (NOT hardware configuration — stays here on purpose).
BRIDGE_CPU_CYCLES = 12.0
BRIDGE_PROVENANCE = (
    "ASSUMPTION (uncalibrated): AXIM->FMC bridge + RPIPE0 capture, ~6 AXI "
    "cycles @200 MHz; bounded by the AN4891 Table-6 cross-check; refine on "
    "hardware"
)

# libDaisy's shipped timing set, kept ONLY for the comparison row in the
# report. This is libDaisy's fact (sdram.cpp as shipped), not this repo's
# configuration — the repo's chosen set lives in daisy_seed_sdram.h and is
# what daisy_spectral_sdram_apply_timings() programs.
STOCK_LIBDAISY_TICKS = {
    "TRP": 16, "TRCD": 10, "TRAS": 4, "TRC": 8, "TWR": 3, "CAS": 3,
}
STOCK_PROVENANCE = (
    "libDaisy sdram.cpp as shipped (RPDelay=16, RCDDelay=10, "
    "SelfRefreshTime=4 — debug-era values; TRAS=4 ticks is 40 ns, below the "
    "48 ns minimum its own comment quotes)"
)

# Empirical anchor for sanity, not a model input: with code in ITCM and
# D-cache ON, AN4891 measured FFT R/W data in SDRAM at 1.19x the DTCM-resident
# time (Table 6, STM32H743I-EVAL @400 MHz).
AN4891_TABLE6_SDRAM_OVER_DTCM = 1.19


class MemoryModelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelConstants:
    """Hardware configuration parsed from the SSOTs + the model assumption."""
    defines: dict[str, int]          # SPECTRAL_DAISY_* from the header
    marker_addr: int                 # from the rig ldscript MARKER region
    header_relpath: str = SDRAM_HEADER_RELPATH

    @property
    def line_bytes(self) -> int:
        return self.defines["SPECTRAL_DAISY_CACHE_LINE"]

    @property
    def sdclk_per_cpu_cycle(self) -> float:
        return self.defines["SPECTRAL_DAISY_CPU_HZ"] / self.defines["SPECTRAL_DAISY_SDCLK_HZ"]

    @property
    def linefill_beats(self) -> float:
        return self.line_bytes * 8 / self.defines["SPECTRAL_DAISY_SDRAM_BUS_BITS"]

    def _fill_cycles(self, cas: int, row_extra_ticks: int, row_hit: bool) -> float:
        ticks = cas + self.linefill_beats + (0 if row_hit else row_extra_ticks)
        return BRIDGE_CPU_CYCLES + ticks * self.sdclk_per_cpu_cycle

    def line_fill_cycles(self, row_hit: bool) -> float:
        """CPU cycles per 32B linefill under the CHOSEN timing set."""
        d = self.defines
        return self._fill_cycles(
            d["SPECTRAL_DAISY_SDRAM_CAS"],
            d["SPECTRAL_DAISY_SDRAM_TRP"] + d["SPECTRAL_DAISY_SDRAM_TRCD"],
            row_hit,
        )

    def line_fill_cycles_stock(self, row_hit: bool) -> float:
        """Same, under libDaisy's shipped timings (comparison row only)."""
        s = STOCK_LIBDAISY_TICKS
        return self._fill_cycles(s["CAS"], s["TRP"] + s["TRCD"], row_hit)

    def writeback_cycles(self) -> float:
        return (self.defines["SPECTRAL_DAISY_SDRAM_TWR"] + self.linefill_beats) \
            * self.sdclk_per_cpu_cycle

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": f"parsed: {self.header_relpath} (per-value provenance "
                      "lives next to each value in that header)",
            "defines": dict(self.defines),
            "marker_addr": hex(self.marker_addr),
            "bridge_cpu_cycles": {"value": BRIDGE_CPU_CYCLES,
                                  "provenance": BRIDGE_PROVENANCE},
            "stock_libdaisy_ticks": {"values": dict(STOCK_LIBDAISY_TICKS),
                                     "provenance": STOCK_PROVENANCE},
        }


def load_constants(repo_root: Path | None = None) -> ModelConstants:
    """Parse the SSOTs. Raises MemoryModelError when the contract moved —
    never silently substitutes a default."""
    root = repo_root or find_repo_root(Path(__file__))

    header = root / SDRAM_HEADER_RELPATH
    try:
        text = header.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryModelError(f"cannot read {SDRAM_HEADER_RELPATH}: {exc}") from exc
    defines = {m.group(1): int(m.group(2)) for m in _DEFINE_RE.finditer(text)}
    missing = [k for k in _REQUIRED_DEFINES if k not in defines]
    if missing:
        raise MemoryModelError(
            f"{SDRAM_HEADER_RELPATH} is missing {missing} — the constants "
            "contract moved; fix the parse or the header, do not hardcode"
        )

    ldscript = root / LDSCRIPT_RELPATH
    try:
        ld_text = ldscript.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryModelError(f"cannot read {LDSCRIPT_RELPATH}: {exc}") from exc
    marker = _MARKER_RE.search(ld_text)
    if marker is None:
        raise MemoryModelError(
            f"MARKER region not found in {LDSCRIPT_RELPATH} — the trace "
            "block-marker contract moved"
        )

    return ModelConstants(defines=defines, marker_addr=int(marker.group(1), 16))


@dataclass(slots=True)
class _Access:
    is_store: bool
    addr: int
    size: int


def parse_trace(path: Path) -> Iterable[_Access]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            kind, addr_hex, size = line.split()
            yield _Access(kind == "S", int(addr_hex, 16), int(size))


class DCacheSim:
    """4-way, write-back write-allocate D-cache (geometry from the SSOT).

    LRU replacement: the TRM does not document the data-cache policy; for the
    segment stream (sequential, compulsory-miss dominated) the policy choice
    is immaterial, which is why the assumption is acceptable — revisit if a
    reuse-heavy SDRAM working set ever approaches the cache size.
    Dynamic read allocate mode [TRM 5.9.1] is approximated: a store that
    writes a full line which is not resident allocates WITHOUT a linefill
    read when it follows full-line-write streaks (memset-class traffic).
    """

    def __init__(self, constants: ModelConstants) -> None:
        self.line_bytes = constants.line_bytes
        self.ways = constants.defines["SPECTRAL_DAISY_DCACHE_WAYS"]
        n_lines = constants.defines["SPECTRAL_DAISY_DCACHE_BYTES"] // self.line_bytes
        self.n_sets = n_lines // self.ways
        self.sets: list[list[tuple[int, bool]]] = [[] for _ in range(self.n_sets)]  # (tag, dirty), MRU first
        self.full_line_write_streak = 0
        # results
        self.read_hits = 0
        self.read_misses = 0
        self.write_hits = 0
        self.write_miss_fills = 0
        self.write_miss_no_fill = 0   # dynamic read allocate mode
        self.dirty_evictions = 0

    def _lookup(self, line_addr: int) -> tuple[int, int]:
        set_idx = line_addr % self.n_sets
        tag = line_addr // self.n_sets
        return set_idx, tag

    def _touch(self, s: list[tuple[int, bool]], i: int, dirty: bool) -> None:
        tag, was_dirty = s.pop(i)
        s.insert(0, (tag, was_dirty or dirty))

    def _fill(self, s: list[tuple[int, bool]], tag: int, dirty: bool) -> None:
        if len(s) >= self.ways:
            _, evict_dirty = s.pop()
            if evict_dirty:
                self.dirty_evictions += 1
        s.insert(0, (tag, dirty))

    def access(self, a: _Access) -> None:
        first = a.addr // self.line_bytes
        last = (a.addr + a.size - 1) // self.line_bytes
        for line_addr in range(first, last + 1):
            set_idx, tag = self._lookup(line_addr)
            s = self.sets[set_idx]
            hit = next((i for i, (t, _) in enumerate(s) if t == tag), None)
            if a.is_store:
                full_line = a.size >= self.line_bytes and a.addr % self.line_bytes == 0
                if hit is not None:
                    self.write_hits += 1
                    self._touch(s, hit, dirty=True)
                elif full_line or self.full_line_write_streak >= 3:
                    self.write_miss_no_fill += 1
                    self._fill(s, tag, dirty=True)
                else:
                    self.write_miss_fills += 1
                    self._fill(s, tag, dirty=True)
                self.full_line_write_streak = (
                    self.full_line_write_streak + 1 if full_line else 0)
            else:
                if hit is not None:
                    self.read_hits += 1
                    self._touch(s, hit, dirty=False)
                else:
                    self.read_misses += 1
                    self._fill(s, tag, dirty=False)


class RowTracker:
    """Open-row tracker per internal SDRAM bank ([row][bank][col] FMC mapping;
    for the sequential segment stream the mapping detail only sets the
    row-miss cadence to one per row_bytes, which dominates either way)."""

    def __init__(self, constants: ModelConstants) -> None:
        self.banks = constants.defines["SPECTRAL_DAISY_SDRAM_BANKS"]
        self.row_bytes = constants.defines["SPECTRAL_DAISY_SDRAM_ROW_BYTES"]
        self.open_rows: dict[int, int] = {}
        self.row_hits = 0
        self.row_misses = 0

    def access(self, addr: int) -> bool:
        lane = addr // self.row_bytes
        bank = lane % self.banks
        row = lane // self.banks
        if self.open_rows.get(bank) == row:
            self.row_hits += 1
            return True
        self.open_rows[bank] = row
        self.row_misses += 1
        return False


def _region(addr: int, marker_addr: int) -> str:
    if addr == marker_addr:
        return "marker"
    if 0x20000000 <= addr < 0x30000000:
        return "dtcm"
    if 0x60000000 <= addr < 0x70000000:
        return "sdram"
    if addr < 0x20000000:
        return "flash"
    return "other"


@dataclass(slots=True)
class BlockStalls:
    constants: ModelConstants
    sdram_accesses: int = 0
    dtcm_accesses: int = 0
    flash_accesses: int = 0
    fills_row_hit: int = 0
    fills_row_miss: int = 0
    write_no_fill: int = 0
    dirty_evictions: int = 0

    @property
    def stall_serial(self) -> float:
        """Upper bound: every miss latency fully exposed."""
        c = self.constants
        return (self.fills_row_hit * c.line_fill_cycles(True)
                + self.fills_row_miss * c.line_fill_cycles(False)
                + (self.dirty_evictions + self.write_no_fill) * c.writeback_cycles())

    @property
    def stall_bandwidth(self) -> float:
        """Lower bound: transfers limited only by SDRAM beat bandwidth
        (two linefill buffers + store buffer hide latency [TRM])."""
        c = self.constants
        lines = (self.fills_row_hit + self.fills_row_miss
                 + self.dirty_evictions + self.write_no_fill)
        return lines * c.linefill_beats * c.sdclk_per_cpu_cycle

    def as_dict(self) -> dict[str, Any]:
        return {
            "accesses": {"dtcm": self.dtcm_accesses, "sdram": self.sdram_accesses,
                         "flash": self.flash_accesses},
            "linefills": {"row_hit": self.fills_row_hit, "row_miss": self.fills_row_miss},
            "writebacks": self.dirty_evictions + self.write_no_fill,
            "stall_cycles": {"bandwidth_bound": round(self.stall_bandwidth, 1),
                             "serial_bound": round(self.stall_serial, 1)},
        }


@dataclass(frozen=True, slots=True)
class MemoryReport:
    constants: ModelConstants
    blocks: tuple[BlockStalls, ...]
    cache: dict[str, int]
    total: BlockStalls

    def as_dict(self) -> dict[str, Any]:
        c = self.constants
        per_block = [b.as_dict() for b in self.blocks]
        worst = max(self.blocks, key=lambda b: b.stall_serial) if self.blocks else None
        return {
            "provenance": "modeled: analytical memory layer over [measured: qemu-tcg] "
                          "trace; hardware constants parsed from the C SSOT "
                          "(daisy_seed_sdram.h); stalls are bounds, not points",
            "constants": c.as_dict(),
            "derived_per_line_cycles": {
                "timing_set": "chosen (daisy_seed_sdram.h — what the firmware programs)",
                "linefill_row_hit": c.line_fill_cycles(True),
                "linefill_row_miss": c.line_fill_cycles(False),
                "writeback": c.writeback_cycles(),
                "stock_libdaisy_comparison": {
                    "linefill_row_hit": c.line_fill_cycles_stock(True),
                    "linefill_row_miss": c.line_fill_cycles_stock(False),
                },
            },
            "an4891_anchor": {
                "table6_sdram_over_dtcm": AN4891_TABLE6_SDRAM_OVER_DTCM,
                "role": "order-of-magnitude cross-check only (FFT workload, not ours)",
            },
            "dcache": self.cache,
            "n_blocks": len(self.blocks),
            "total": self.total.as_dict(),
            "per_block_mean_stalls": {
                "bandwidth_bound": round(sum(b.stall_bandwidth for b in self.blocks)
                                         / max(1, len(self.blocks)), 1),
                "serial_bound": round(sum(b.stall_serial for b in self.blocks)
                                      / max(1, len(self.blocks)), 1),
            },
            "worst_block": worst.as_dict() if worst else None,
            "per_block": per_block,
        }


def analyze_trace(
    trace_path: Path, constants: ModelConstants | None = None
) -> MemoryReport:
    """Replay the trace; split into blocks on the marker store; price SDRAM-class
    misses. DTCM-class accesses are counted but never stall [TRM/AN4891]."""
    c = constants or load_constants()
    cache = DCacheSim(c)
    rows = RowTracker(c)
    blocks: list[BlockStalls] = []
    cur = BlockStalls(constants=c)
    total = BlockStalls(constants=c)

    def fills_snapshot() -> tuple[int, int, int]:
        return (cache.read_misses + cache.write_miss_fills,
                cache.write_miss_no_fill, cache.dirty_evictions)

    prev_fills, prev_nofill, prev_evict = fills_snapshot()

    for a in parse_trace(trace_path):
        region = _region(a.addr, c.marker_addr)
        if region == "marker":
            blocks.append(cur)
            cur = BlockStalls(constants=c)
            continue
        if region == "dtcm":
            cur.dtcm_accesses += 1
            total.dtcm_accesses += 1
            continue
        if region == "flash":
            cur.flash_accesses += 1
            total.flash_accesses += 1
            continue
        if region != "sdram":
            continue
        cur.sdram_accesses += 1
        total.sdram_accesses += 1
        cache.access(a)
        fills, nofill, evict = fills_snapshot()
        new_fills = fills - prev_fills
        if new_fills:
            for _ in range(new_fills):
                if rows.access(a.addr):
                    cur.fills_row_hit += 1
                    total.fills_row_hit += 1
                else:
                    cur.fills_row_miss += 1
                    total.fills_row_miss += 1
        cur.write_no_fill += nofill - prev_nofill
        total.write_no_fill += nofill - prev_nofill
        cur.dirty_evictions += evict - prev_evict
        total.dirty_evictions += evict - prev_evict
        prev_fills, prev_nofill, prev_evict = fills, nofill, evict

    if cur.sdram_accesses or cur.dtcm_accesses or cur.flash_accesses:
        blocks.append(cur)

    cache_stats = {
        "read_hits": cache.read_hits, "read_misses": cache.read_misses,
        "write_hits": cache.write_hits, "write_miss_fills": cache.write_miss_fills,
        "write_miss_no_fill": cache.write_miss_no_fill,
        "dirty_evictions": cache.dirty_evictions,
        "sets": cache.n_sets, "ways": cache.ways, "line_bytes": cache.line_bytes,
    }
    return MemoryReport(constants=c, blocks=tuple(blocks), cache=cache_stats, total=total)
