"""Generate or verify a C source file containing resource file hashes.

This tool scans a directory recursively and emits one global array of
(path, hash, size) entries plus a count symbol, using Python xxHash.

- simple O(n) traversal/search by callers
- deterministic ordering

Usage (generate):
    PYTHONPATH=tools python3 -m spectral_tools.generators.resource_hashes \
        --resources-dir resources \
        --output spectral_engine/core/spectral_hash_resources_xx32_xx3.c

Usage (verify):
    PYTHONPATH=tools python3 -m spectral_tools.generators.resource_hashes \
        --resources-dir resources \
        --output spectral_engine/core/spectral_hash_resources_xx32_xx3.c \
        --mode verify
"""

from __future__ import annotations

import argparse
import re
import sys

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# When executed directly (python resource_hashes.py), __package__ is unset.
# Patch sys.path once so relative imports work in both modes.
if __package__ in {None, ""}:
    _tools_root = str(Path(__file__).resolve().parents[2])
    if _tools_root not in sys.path:
        sys.path.insert(0, _tools_root)
    __package__ = "spectral_tools.generators"  # noqa: A001

from ..core.cli import add_dry_run_flag, add_print_branch_flag
from ..core.console import Console
from ..core.constants import DEFAULT_UTF8_ENCODING
from ..core.utils import (
    list_files_recursive,
    list_repo_files_or_rglob,
    read_utf8_lf,
    write_utf8_lf,
)
from . import native_bridge as _bridge

import xxhash  # type: ignore


# FNV-1a 32-bit constants — kept for decode_fnv1a32_offset_basis() only.
# The forward hash is now computed by the native C bridge.
FNV1A_32_OFFSET_BASIS = 2166136261
FNV1A_32_PRIME = 16777619
FNV1A_32_MASK = 0xFFFFFFFF
FNV1A_32_MODULUS = 1 << 32
FNV1A_32_PRIME_INVERSE = pow(FNV1A_32_PRIME, -1, FNV1A_32_MODULUS)
FILE_READ_CHUNK_BYTES = 8192
DEFAULT_RESOURCES_DIR = Path("resources")
DEFAULT_OUTPUT_PATH = Path("spectral_engine/core/spectral_hash_resources_xx32_xx3.c")
DEFAULT_MODE = "generate"
MODE_CHOICES = ("generate", "verify")
DEFAULT_FORCE_RGLOB = False
DEFAULT_PRINT_BRANCH = False
DEFAULT_DRY_RUN = False
HOST_ROW_PATTERN = re.compile(
    r'^\s*\{\s*"(?P<path>(?:\\.|[^"\\])*)"\s*,\s*'
    r'UINT64_C\(0x(?P<hash>[0-9a-fA-F]{1,16})\)\s*,\s*'
    r'\(size_t\)(?P<size>\d+)u\s*\},\s*$'
)
EMBEDDED_ROW_PATTERN = re.compile(
    r'^\s*\{\s*UINT32_C\(0x(?P<id>[0-9a-fA-F]{1,8})\)\s*,\s*'
    r'UINT64_C\(0x(?P<hash>[0-9a-fA-F]{1,16})\)\s*,\s*'
    r'\(size_t\)(?P<size>\d+)u\s*\},\s*$'
)



class HashAlgorithm(str, Enum):
    XXH3_64 = "xxh3_64"
    XXH32 = "xxh32"


@dataclass(frozen=True)
class ResourceHashEntry:
    relative_path: str
    canonical_path: bytes
    file_id: int
    host_digest: int
    embedded_digest: int
    size_bytes: int


@dataclass(frozen=True)
class EntryCheckResult:
    entry: ResourceHashEntry
    status: str
    host_equivalent: bool
    embedded_equivalent: bool
    host_expected_tuple: str
    host_found_tuple: str
    embedded_expected_tuple: str
    embedded_found_tuple: str
    embedded_expected_decode: str


@dataclass(frozen=True)
class ParsedHostRow:
    path_escaped: str
    digest: int
    size_bytes: int


@dataclass(frozen=True)
class ParsedEmbeddedRow:
    file_id: int
    digest: int
    size_bytes: int


def file_id_from_path(canonical_path: bytes) -> int:
    """Compute FNV-1a (32-bit) file ID via the native C bridge.

    Callers must pass the result of :func:`compress_path`, not a raw path.
    """
    return _bridge.file_id_from_path(canonical_path)


def decode_fnv1a32_offset_basis(canonical_path: bytes, digest: int) -> int:
    """Reverse FNV-1a over known bytes to recover the initial offset basis."""
    value = digest & FNV1A_32_MASK
    for byte in reversed(canonical_path):
        value = (value * FNV1A_32_PRIME_INVERSE) & FNV1A_32_MASK
        value ^= byte
    return value


def compress_path(path: Path, resources_dir: Path) -> bytes:
    """Canonicalize a resource-relative path via the native C bridge."""
    relative_posix = path.relative_to(resources_dir).as_posix()
    return _bridge.compress_path(relative_posix)


def hash_file_xxhash(path: Path, algorithm: HashAlgorithm) -> int:
    if algorithm == HashAlgorithm.XXH3_64:
        hasher = xxhash.xxh3_64()
    elif algorithm == HashAlgorithm.XXH32:
        hasher = xxhash.xxh32()
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(FILE_READ_CHUNK_BYTES), b""):
            hasher.update(chunk)

    return hasher.intdigest()


def c_escape(value: str) -> str:
    """Escape UTF-8 bytes for a C string literal using ASCII source text only."""
    def _escape_byte(b: int) -> str:
        if b == 0x5C:  # '\'
            return '\\\\'
        if b == 0x22:  # '"'
            return '\\"'
        if b == 0x09:  # '\t'
            return '\\t'
        if b == 0x0A:  # '\n'
            return '\\n'
        if b == 0x0D:  # '\r'
            return '\\r'
        if 0x20 <= b <= 0x7E:
            return chr(b)
        return f'\\x{b:02x}'
    return ''.join(_escape_byte(b) for b in value.encode(DEFAULT_UTF8_ENCODING))


def list_resource_files(resources_dir: Path, *, force_rglob: bool = DEFAULT_FORCE_RGLOB) -> list[Path]:
    if force_rglob:
        # Explicit override: ignore git-tracked filtering and scan the directory.
        return list_files_recursive(resources_dir)
    return list_repo_files_or_rglob(resources_dir)


def collect_entries(
    resources_dir: Path,
    *,
    force_rglob: bool = DEFAULT_FORCE_RGLOB,
) -> list[ResourceHashEntry]:
    files = list_resource_files(resources_dir, force_rglob=force_rglob)

    entries: list[ResourceHashEntry] = []
    for path in files:
        rel = path.relative_to(resources_dir).as_posix()
        canonical = compress_path(path, resources_dir)
        host_hash = hash_file_xxhash(path, HashAlgorithm.XXH3_64)
        embedded_hash = hash_file_xxhash(path, HashAlgorithm.XXH32)
        file_size = path.stat().st_size
        entries.append(ResourceHashEntry(
            relative_path=rel,
            canonical_path=canonical,
            file_id=file_id_from_path(canonical),
            host_digest=host_hash,
            embedded_digest=embedded_hash,
            size_bytes=file_size,
        ))
    return entries


def _embedded_row(entry: ResourceHashEntry) -> str:
    return (
        f"    {{ UINT32_C(0x{entry.file_id:08x}), "
        f"UINT64_C(0x{entry.embedded_digest:016x}), (size_t){entry.size_bytes}u }},"
    )


def _host_row(entry: ResourceHashEntry) -> str:
    return (
        f'    {{"{c_escape(entry.relative_path)}", '
        f"UINT64_C(0x{entry.host_digest:016x}), (size_t){entry.size_bytes}u}},"
    )


def _format_host_tuple(path_escaped: str, digest: int, size_bytes: int) -> str:
    return f'("{path_escaped}", 0x{digest:016x}, {size_bytes})'


def _format_embedded_tuple(file_id: int, digest: int, size_bytes: int) -> str:
    return f"(0x{file_id:08x}, 0x{digest:016x}, {size_bytes})"


def _parse_output_rows(existing_content: str) -> tuple[dict[str, ParsedHostRow], dict[int, ParsedEmbeddedRow]]:
    host_rows: dict[str, ParsedHostRow] = {}
    embedded_rows: dict[int, ParsedEmbeddedRow] = {}

    for line in existing_content.splitlines():
        host_match = HOST_ROW_PATTERN.match(line)
        if host_match:
            path_escaped = host_match.group("path")
            host_kwargs = {
                "path_escaped": path_escaped,
                "digest": int(host_match.group("hash"), 16),
                "size_bytes": int(host_match.group("size")),
            }
            host_rows[path_escaped] = ParsedHostRow(**host_kwargs)
            continue

        embedded_match = EMBEDDED_ROW_PATTERN.match(line)
        if embedded_match:
            file_id = int(embedded_match.group("id"), 16)
            embedded_kwargs = {
                "file_id": file_id,
                "digest": int(embedded_match.group("hash"), 16),
                "size_bytes": int(embedded_match.group("size")),
            }
            embedded_rows[file_id] = ParsedEmbeddedRow(**embedded_kwargs)

    return host_rows, embedded_rows


def build_entry_check_results(entries: list[ResourceHashEntry], existing_content: str | None) -> list[EntryCheckResult]:
    host_rows: dict[str, ParsedHostRow] = {}
    embedded_rows: dict[int, ParsedEmbeddedRow] = {}
    if existing_content is not None:
        host_rows, embedded_rows = _parse_output_rows(existing_content)

    results: list[EntryCheckResult] = []
    for entry in entries:
        path_escaped = c_escape(entry.relative_path)
        decode_basis = decode_fnv1a32_offset_basis(entry.canonical_path, entry.file_id)
        decode_matches = decode_basis == FNV1A_32_OFFSET_BASIS
        embedded_expected_decode = (
            f"basis=0x{decode_basis:08x} "
            f"matches_offset_basis={str(decode_matches).lower()}"
        )

        expected_host_tuple = _format_host_tuple(
            path_escaped,
            entry.host_digest,
            entry.size_bytes,
        )
        expected_embedded_tuple = _format_embedded_tuple(
            entry.file_id,
            entry.embedded_digest,
            entry.size_bytes,
        )

        if existing_content is None:
            found_host_tuple = expected_host_tuple
            found_embedded_tuple = expected_embedded_tuple
            host_equivalent = True
            embedded_equivalent = True
        else:
            found_host = host_rows.get(path_escaped)
            found_embedded = embedded_rows.get(entry.file_id)

            if found_host is None:
                found_host_tuple = "<missing>"
                host_equivalent = False
            else:
                found_host_tuple = _format_host_tuple(
                    found_host.path_escaped,
                    found_host.digest,
                    found_host.size_bytes,
                )
                host_equivalent = (
                    found_host.digest == entry.host_digest
                    and found_host.size_bytes == entry.size_bytes
                )

            if found_embedded is None:
                found_embedded_tuple = "<missing>"
                embedded_equivalent = False
            else:
                found_embedded_tuple = _format_embedded_tuple(
                    found_embedded.file_id,
                    found_embedded.digest,
                    found_embedded.size_bytes,
                )
                embedded_equivalent = (
                    found_embedded.digest == entry.embedded_digest
                    and found_embedded.size_bytes == entry.size_bytes
                )

        status = "ok" if host_equivalent and embedded_equivalent else "failed"
        result_kwargs = {
            "entry": entry,
            "status": status,
            "host_equivalent": host_equivalent,
            "embedded_equivalent": embedded_equivalent,
            "host_expected_tuple": expected_host_tuple,
            "host_found_tuple": found_host_tuple,
            "embedded_expected_tuple": expected_embedded_tuple,
            "embedded_found_tuple": found_embedded_tuple,
            "embedded_expected_decode": embedded_expected_decode,
        }
        results.append(EntryCheckResult(**result_kwargs))
    return results


def render_branch_results(
    *,
    mode: str,
    resources_dir: Path,
    output_path: Path,
    force_rglob: bool,
    results: list[EntryCheckResult],
) -> str:
    from ..core.branch_view import BranchFormatter

    tests = []
    for index, item in enumerate(results, start=1):
        entry = item.entry
        if mode == "generate":
            details = {
                "size_bytes": entry.size_bytes,
                "generated": {
                    "line_index": index,
                    "host_row": _host_row(entry).strip(),
                    "embedded_row": _embedded_row(entry).strip(),
                },
            }
            status = "ok"
        else:
            host_detail = {
                "equal": item.host_equivalent,
                "expected": item.host_expected_tuple,
                "found": item.host_found_tuple,
            }
            embedded_detail = {
                "equal": item.embedded_equivalent,
                "expected": item.embedded_expected_tuple,
                "found": item.embedded_found_tuple,
            }
            if mode == "verify":
                embedded_detail["expected_decode"] = item.embedded_expected_decode

            details = {
                "size_bytes": entry.size_bytes,
                "equivalence": {
                    "host": host_detail,
                    "embedded": embedded_detail,
                },
            }
            status = item.status
        test_kwargs = {
            "name": entry.relative_path,
            "status": status,
            "summary": (
                f"id=0x{entry.file_id:08x} "
                f"host=0x{entry.host_digest:016x} "
                f"embedded=0x{entry.embedded_digest:016x}"
            ),
            "details": details,
        }
        tests.append(test_kwargs)

    report = {
        "suite": "resource_hashes",
        "context": {
            "mode": mode,
            "resources_dir": str(resources_dir),
            "output": str(output_path),
            "force_rglob": force_rglob,
            "entry_count": len(results),
        },
        "tests": tests,
    }
    return BranchFormatter(use_color=True).render(report)


def generate_c(entries: list[ResourceHashEntry]) -> str:
    header_lines = [
        "/* AUTO-GENERATED by tools/spectral_tools/generators/resource_hashes.py — do not edit */",
        "/* Hash algorithms: host=xxh3_64, embedded=xxh32 */",
        "/*",
        " * Note:",
        " *   Preprocessor-time file hashing is not supported by C/C11.",
        " *   Build-time equivalence is enforced by CMake verify_resource_hashes target.",
        " *",
        " * Example (commented) static assert style:",
        " *   // _Static_assert(1, \"resource hash equivalence verified in build step\");",
        " */",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include \"spectral_resource_fs.h\"",
        "",
        "const SpectralResourceFsEntry spectral_resource_hashes[] = {",
        "#if SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM",
    ]
    embedded_rows = [_embedded_row(entry) for entry in entries]
    host_rows = [_host_row(entry) for entry in entries]
    footer_lines = [
        "#endif",
        "};",
        "",
        f"const size_t spectral_resource_hashes_count = (size_t){len(entries)}u;",
        "",
    ]
    return "\n".join([*header_lines, *embedded_rows, "#else", *host_rows, *footer_lines])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate C resource hash array")
    parser.add_argument(
        "-r",
        "--resources-dir",
        type=Path,
        default=DEFAULT_RESOURCES_DIR,
        help="Directory to scan recursively (default: resources)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output C source path",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=MODE_CHOICES,
        default=DEFAULT_MODE,
        help="generate: emit C to stdout (and write --output unless --dry-run); verify: compare output against fresh generation",
    )
    parser.add_argument(
        "-R",
        "--force-rglob",
        action="store_true",
        default=DEFAULT_FORCE_RGLOB,
        help=(
            "Bypass git-tracked discovery and hash all files found recursively "
            "under --resources-dir."
        ),
    )
    add_print_branch_flag(parser, default=DEFAULT_PRINT_BRANCH)
    add_dry_run_flag(parser, help_text="Generate hashes without writing --output (generate mode only).")
    args = parser.parse_args()
    console = Console()
    if args.mode != "generate" and args.dry_run:
        console.print_error("--dry-run is only valid with --mode generate")
        return 1

    resources_dir = args.resources_dir.resolve()
    if not resources_dir.exists() or not resources_dir.is_dir():
        console.print_error(f"resources directory not found: {resources_dir}")
        return 1

    entries = collect_entries(resources_dir, force_rglob=args.force_rglob)
    content = generate_c(entries)
    branch_payload: str | None = None

    if args.mode == "generate":
        if args.print_branch:
            results = build_entry_check_results(entries, existing_content=None)
            branch_payload = render_branch_results(
                mode=args.mode,
                resources_dir=resources_dir,
                output_path=args.output,
                force_rglob=args.force_rglob,
                results=results,
            )
        print(content)
        if args.dry_run:
            if branch_payload:
                print()
                print(branch_payload)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_utf8_lf(args.output, content)
        if branch_payload:
            print()
            print(branch_payload)
        return 0

    print("Using Python xxhash module (xxh3_64 host, xxh32 embedded)")
    if args.force_rglob:
        print("Resource discovery: forced recursive scan (git ignored)")
    else:
        print("Resource discovery: tracked files when available (fallback: recursive scan)")

    if not args.output.exists():
        if args.print_branch:
            results = build_entry_check_results(entries, existing_content="")
            print(
                render_branch_results(
                    mode=args.mode,
                    resources_dir=resources_dir,
                    output_path=args.output,
                    force_rglob=args.force_rglob,
                    results=results,
                )
            )
        console.print_error(f"verify failed: output file does not exist: {args.output}")
        return 1

    existing = read_utf8_lf(args.output)
    if args.print_branch:
        branch_payload = render_branch_results(
            mode=args.mode,
            resources_dir=resources_dir,
            output_path=args.output,
            force_rglob=args.force_rglob,
            results=build_entry_check_results(entries, existing),
        )

    if existing != content:
        if branch_payload:
            print(branch_payload)
        console.print_error(
            "verify failed: generated hashes do not match committed/generated output. "
            "Run in generate mode to refresh."
        )
        return 1

    print(f"Verified {args.output}: hashes match {len(entries)} resources")
    if branch_payload:
        print()
        print(branch_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
