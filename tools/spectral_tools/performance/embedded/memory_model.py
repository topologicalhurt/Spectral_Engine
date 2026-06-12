"""Layer 3: analytical memory-stall model over Layer-1 address traces (P4).

Replays the QEMU plugin's data-access trace (program order, exact addresses)
through the Daisy placement and a Cortex-M7 D-cache model, pricing misses with
a latency table whose every constant carries its calibration provenance —
the shipped libDaisy FMC/clock configuration, the M7 TRM structure facts, and
the AN4891 measured anchors. Nothing here is a measurement of hardware: the
trace is measured, the stalls are MODELED, and the report says which is which
(M7_PERF_MODEL_PLAN fidelity contract).

Placement mirror (the counts rig's ldscript ⇄ Daisy):
  DATA  0x20000000  -> DTCM-class   (Daisy .dtcmram_bss: ctx, q63 accum, blk)
  BULK  0x60000000  -> SDRAM-class  (Daisy .sdram_bss: the segment store),
                       reached through the 16 KB 4-way WBWA D-cache
  CODE  <0x20000000 -> flash-class  (rodata via D-cache; code via I-cache)

Stall accounting is reported as a RANGE, not a point: the M7 is non-blocking
(two data linefill buffers, merging store buffer [TRM]), so the true cost of N
misses lies between the bandwidth bound (transfers fully overlapped) and the
serial bound (every miss latency exposed). A point estimate would be theater.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

LINE_BYTES = 32          # [TRM DDI 0489F 5.9: "Both caches use a line-length of 32-bytes"]
MARKER_ADDR = 0x60F00000  # block boundary marker (rig ldscript .bulk_marker)


@dataclass(frozen=True, slots=True)
class Constant:
    value: float
    provenance: str


# Every model constant with its calibration provenance. CPU cycles @ 400 MHz
# (the libDaisy default clock) unless stated otherwise.
LATENCY = {
    "cpu_hz": Constant(400e6, "libDaisy system.cpp FREQ_400MHZ default (PLLN=200, HSE)"),
    "dtcm_stall_cycles": Constant(0.0,
        "TRM DDI 0489F: internal TCM zero-wait; AN4891 Rev1 p17: DTCM 'accessible at the "
        "maximum CPU clock speed (400 MHz) without latency'"),
    "sdclk_per_cpu_cycle": Constant(4.0,
        "derived: FMC kernel = PLL2R 200 MHz (libDaisy system.cpp), SDClockPeriod=2 "
        "(libDaisy sdram.cpp) -> SDCLK 100 MHz vs CPU 400 MHz"),
    "linefill_beats": Constant(8.0,
        "derived: 32B line [TRM 5.9] / 32-bit SDRAM data bus "
        "(libDaisy FMC_SDRAM_MEM_BUS_WIDTH_32); RBURST enabled, RPIPE 0"),
    "cas_sdclk": Constant(3.0, "libDaisy sdram.cpp FMC_SDRAM_CAS_LATENCY_3"),
    "row_open_extra_sdclk": Constant(26.0,
        "libDaisy sdram.cpp AS SHIPPED: RPDelay=16 + RCDDelay=10 (comments say "
        "'started at 2'; AS4C16M32MSA-6 datasheet minimum is ~3+3 at 100 MHz — the "
        "shipped config is ~4x conservative; the model prices the device as configured)"),
    "writeback_extra_sdclk": Constant(3.0, "libDaisy sdram.cpp WriteRecoveryTime=3"),
    "bridge_cpu_cycles": Constant(12.0,
        "ASSUMPTION (uncalibrated): AXIM->FMC bridge + RPIPE0 capture, ~6 AXI cycles "
        "@200 MHz; bounded by the AN4891 Table-6 cross-check below; refine on hardware"),
    "row_bytes": Constant(2048.0,
        "AS4C16M32MSA organization: 512 columns x 32-bit = 2 KB per row per bank"),
    "banks": Constant(4.0, "libDaisy sdram.cpp FMC_SDRAM_INTERN_BANKS_NUM_4"),
    "dcache_bytes": Constant(16384.0,
        "AN4891 Rev1 p5 Table: STM32H7x3 data cache 16 Kbytes"),
    "dcache_ways": Constant(4.0, "TRM DDI 0489F 5.9: data cache four-way set-associative"),
}

# Empirical anchor for sanity, not a model input: with code in ITCM and D-cache
# ON, AN4891 measured FFT R/W data in SDRAM at 1.19x the DTCM-resident time
# (Table 6, STM32H743I-EVAL @400 MHz). A modeled SDRAM stall share wildly
# inconsistent with that order of magnitude means a constant is wrong.
AN4891_TABLE6_SDRAM_OVER_DTCM = 1.19

# Derived per-line costs (CPU cycles).
def line_fill_cycles(row_hit: bool) -> float:
    sd = LATENCY["sdclk_per_cpu_cycle"].value
    cyc = LATENCY["bridge_cpu_cycles"].value + (
        LATENCY["cas_sdclk"].value + LATENCY["linefill_beats"].value) * sd
    if not row_hit:
        cyc += LATENCY["row_open_extra_sdclk"].value * sd
    return cyc


def writeback_cycles() -> float:
    sd = LATENCY["sdclk_per_cpu_cycle"].value
    return (LATENCY["writeback_extra_sdclk"].value + LATENCY["linefill_beats"].value) * sd


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
    """16 KB, 4-way, 32B-line, write-back write-allocate D-cache.

    LRU replacement: the TRM does not document the data-cache policy; for the
    segment stream (sequential, compulsory-miss dominated) the policy choice
    is immaterial, which is why the assumption is acceptable — revisit if a
    reuse-heavy SDRAM working set ever approaches 16 KB.
    Dynamic read allocate mode [TRM 5.9.1] is approximated: a store that
    writes a full line which is not resident allocates WITHOUT a linefill
    read when it follows full-line-write streaks (memset-class traffic).
    """

    def __init__(self) -> None:
        n_lines = int(LATENCY["dcache_bytes"].value) // LINE_BYTES
        self.ways = int(LATENCY["dcache_ways"].value)
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
        first = a.addr // LINE_BYTES
        last = (a.addr + a.size - 1) // LINE_BYTES
        for line_addr in range(first, last + 1):
            set_idx, tag = self._lookup(line_addr)
            s = self.sets[set_idx]
            hit = next((i for i, (t, _) in enumerate(s) if t == tag), None)
            if a.is_store:
                full_line = a.size >= LINE_BYTES and a.addr % LINE_BYTES == 0
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

    def __init__(self) -> None:
        self.banks = int(LATENCY["banks"].value)
        self.row_bytes = int(LATENCY["row_bytes"].value)
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


def _region(addr: int) -> str:
    if addr == MARKER_ADDR:
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
        return (self.fills_row_hit * line_fill_cycles(True)
                + self.fills_row_miss * line_fill_cycles(False)
                + (self.dirty_evictions + self.write_no_fill) * writeback_cycles())

    @property
    def stall_bandwidth(self) -> float:
        """Lower bound: transfers limited only by SDRAM beat bandwidth
        (two linefill buffers + store buffer hide latency [TRM])."""
        sd = LATENCY["sdclk_per_cpu_cycle"].value
        lines = (self.fills_row_hit + self.fills_row_miss
                 + self.dirty_evictions + self.write_no_fill)
        return lines * LATENCY["linefill_beats"].value * sd

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
    blocks: tuple[BlockStalls, ...]
    cache: dict[str, int]
    total: BlockStalls

    def as_dict(self) -> dict[str, Any]:
        per_block = [b.as_dict() for b in self.blocks]
        worst = max(self.blocks, key=lambda b: b.stall_serial) if self.blocks else None
        return {
            "provenance": "modeled: analytical memory layer over [measured: qemu-tcg] "
                          "trace; constants from libDaisy-as-shipped + TRM + AN4891 "
                          "(see constants); stalls are bounds, not points",
            "constants": {k: {"value": c.value, "provenance": c.provenance}
                          for k, c in LATENCY.items()},
            "derived_per_line_cycles": {
                "linefill_row_hit": line_fill_cycles(True),
                "linefill_row_miss": line_fill_cycles(False),
                "writeback": writeback_cycles(),
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


def analyze_trace(trace_path: Path) -> MemoryReport:
    """Replay the trace; split into blocks on the marker store; price SDRAM-class
    misses. DTCM-class accesses are counted but never stall [TRM/AN4891]."""
    cache = DCacheSim()
    rows = RowTracker()
    blocks: list[BlockStalls] = []
    cur = BlockStalls()
    total = BlockStalls()

    def fills_snapshot() -> tuple[int, int, int]:
        return (cache.read_misses + cache.write_miss_fills,
                cache.write_miss_no_fill, cache.dirty_evictions)

    prev_fills, prev_nofill, prev_evict = fills_snapshot()

    for a in parse_trace(trace_path):
        region = _region(a.addr)
        if region == "marker":
            blocks.append(cur)
            cur = BlockStalls()
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
        "sets": cache.n_sets, "ways": cache.ways, "line_bytes": LINE_BYTES,
    }
    return MemoryReport(blocks=tuple(blocks), cache=cache_stats, total=total)
