# Patch notes — Pass 182: CTF sweep increment 22 — host file-I/O layer (spectral_fs.c + spectral_seg_cache_fs.c, clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **host filesystem
primitives** that back the segment cache (Pass 180) — the layer that actually opens,
sizes, seeks, reads, mmaps, and appends bytes, where offset arithmetic, page-alignment,
and EOF bounds errors would live:

```text
- core/spectral_fs.c              u64<->off_t conversion, open/open_optional/close,
                                  seek/tell/file_size, read_exact / read_exact_path,
                                  write_exact, map_ro_path (mmap + page alignment),
                                  unmap
- core/spectral_seg_cache_fs.c    index load/write (header + entry endian swap, exact
                                  file-length validation), data append begin/write/end/
                                  abort, data_file_size, data_read_exact, data_map_ro
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol a
clean audit is a legitimate result and a defect must not be fabricated.

## What was checked and why it is correct

### Every offset+bytes is overflow-guarded, then bounded by the real file size

```text
- spectral_fs_read_exact_path (139-173): opens, file_size, then
  `if (bytes > UINT64_MAX - offset || size < offset + bytes) return FILE_CORRUPT;`
  — the first clause prevents the uint64 add from wrapping, the second rejects a read
  that runs past EOF. Only then does it seek(offset)+read_exact(bytes). A short read
  (fread != bytes) returns FILE_READ.
- spectral_fs_map_ro_path (202-274): fstat, then the identical
  `bytes > UINT64_MAX - offset || st.st_size < offset + bytes` guard before mapping.
```

### mmap page-alignment math cannot overflow and never maps past EOF

```text
- page_off_u64 = offset % page_sz; map_start_u64 = offset - page_off_u64 (page-aligned).
  page_off = (size_t)page_off_u64 is < page_sz so the size_t cast is lossless.
- `if (bytes > SIZE_MAX - page_off)` guards map_len = page_off + bytes against size_t
  overflow; spectral_fs_u64_to_off(map_start_u64, &map_start) rejects an offset that
  does not fit a non-negative off_t. page_sz<=0 (sysconf failure) is rejected.
- The mapped region [map_start, map_start+map_len) = [map_start, offset+bytes). Since
  the bounds check already proved st.st_size >= offset+bytes, the whole mapping is
  backed by file bytes — the consumer reads [base+page_off, +bytes) = file
  [offset, offset+bytes), so no access can fault past EOF (no SIGBUS).
- fd is closed on EVERY path: fstat fail, bounds fail, page_sz fail, overflow fail,
  u64_to_off fail, and unconditionally after mmap (before the MAP_FAILED test) — no
  descriptor leak. On success base/len/page_offset are published; unmap munmaps len.
```

### Index loader validates exact file length before allocating (no corrupt-count blowup)

```text
- spectral_seg_cache_fs_index_load (45-149): reads the header, checks magic + version,
  swaps endian. count==0 requires file_size == sizeof(header) exactly. count>0 computes
  entries_bytes via spectral_array_bytes (overflow-checked), folds with the header size
  via spectral_size_add, and requires file_size == header+entries EXACTLY before the
  malloc — so a corrupt huge count cannot request a giant allocation for a small file.
  read_exact then fills entries_bytes and each entry is endian-swapped. close error is
  folded into the return only when the primary op succeeded.
- index_write mirrors it: header then a swapped temp copy of the entries (the caller's
  buffer is left host-endian); array size is overflow-checked; tmp freed on all paths.
```

### Append is transactional and offset-correct; size/read/seek are 64-bit clean

```text
- data_append_begin opens "ab", seeks END, tells -> data_offset (the authoritative
  append position). append_write delegates to write_exact; append_end closes and
  returns data_offset; append_abort closes and zeroes the writer (so a half-written
  record is abandoned, never indexed — matches the seg_cache store's goto append_error).
- spectral_fs_file_size saves the current position, seeks END, tells, and restores the
  saved position via fseeko/ftello (off_t, 64-bit) — non-destructive to the stream.
- spectral_fs_seek rejects negative offsets; spectral_fs_seek_u64 / _u64_to_off reject
  values that do not round-trip to a non-negative off_t (SPECTRAL_ERR_OVERFLOW).
```

## Verification

```text
- No source changed this pass (read-only audit), so the Pass 181 green state is
  preserved by construction — host binaries remain byte-identical.
  Re-confirmed the gate:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-181 (see prior notes) and now the host
file-I/O layer (182, clean — every offset+bytes is overflow-guarded then bounded by the
actual file size before any seek/read/mmap; the mmap page-offset arithmetic is
overflow-checked and the mapped region is provably within the file so no access faults
past EOF; the fd is closed on every error path; the index loader validates the exact
count-derived file length before allocating, defeating a corrupt-count allocation blowup;
append is transactional and 64-bit-offset clean). Phase D (compiled harness + LUT
golden-vector loop) follows.
