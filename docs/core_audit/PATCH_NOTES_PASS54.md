# Core audit pass 54: wavetable HEX record shape contract

## Summary

Pass 54 hardens the Intel HEX wavetable parser.

Pass 35 made HEX ingestion require EOF, full coverage and subtractive bounds.
The per-line parser still did not validate the declared record length against
the actual line length before parsing fields.

## Bug

`parse_hex_line()` parsed `byte_count`, then used fixed offsets into the line:

```c
line + 9 + (i * 2)
line + 9 + (byte_count * 2)
```

The parser limited `byte_count` by the local data buffer, but it did not prove
the text line actually contained exactly the declared number of data bytes plus
checksum.

The EOF record branch also accepted any type-01 record without enforcing the
Intel HEX EOF shape:

```text
byte_count == 0
address == 0
```

## Fix

`parse_hex_line()` now validates:

```text
line begins with ':'
line length before CR/LF is exactly 11 + 2 * byte_count
byte_count <= data_capacity
all output pointers are present
```

The HEX loader now accepts EOF only when:

```text
record_type == 0x01
data_len == 0
address == 0
```

## Reviewer Walkthrough

1. `line_len` is computed with `strcspn(line, "\r\n")`.
2. Minimal record length is checked before field parsing.
3. `byte_count` is parsed and bounded by the stack data buffer.
4. The exact declared line length is checked before data/checksum parsing.
5. EOF records with payload or nonzero address are rejected.
6. Existing checksum and coverage validation remains.

## Why this is critical

HEX files are text-encoded binary input. The declared byte count is part of the
record shape; accepting malformed record lengths can hide truncated, concatenated
or otherwise corrupted wavetable payloads.
