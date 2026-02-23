#!/usr/bin/env python3
"""Generate or verify a C source file containing resource file hashes.

This tool scans a directory recursively and emits one global array of
(path, hash, size) entries plus a count symbol, using xxHash via xxhsum.

- simple O(n) traversal/search by callers
- deterministic ordering

Usage (generate):
    python3 tools/gen_resource_hashes.py \
        --resources-dir resources \
        --output spectral_engine/core/spectral_resource_hashes_xx32_xx3.c

Usage (verify):
    python3 tools/gen_resource_hashes.py \
        --resources-dir resources \
        --output spectral_engine/core/spectral_resource_hashes_xx32_xx3.c \
        --mode verify
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass

from pathlib import Path

try:
    import xxhash  # type: ignore
except ImportError:
    xxhash = None


# FNV-1a 32-bit constants — must match SPECTRAL_FNV1A_32_OFFSET_BASIS /
# SPECTRAL_FNV1A_32_PRIME in spectral_resource_fs.h.  Any divergence causes
# silent ID mismatches on embedded targets.
_FNV1A_32_OFFSET_BASIS = 2166136261
_FNV1A_32_PRIME = 16777619
_FNV1A_32_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class ResourceHashEntry:
    relative_path: str
    file_id: int
    host_digest: int
    embedded_digest: int
    size_bytes: int


def file_id_from_path(canonical_path: str) -> int:
    """Compute FNV-1a (32-bit) over a canonical path string.

    Callers must pass the result of compress_path(), not the raw relative path,
    so that the ID matches what the C runtime computes via
    spectral_resource_file_id_from_path() in spectral_resource_fs.c.
    """
    digest = _FNV1A_32_OFFSET_BASIS
    for byte in canonical_path.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV1A_32_PRIME) & _FNV1A_32_MASK
    return digest


_RLE_SENTINEL = '\x01'


def compress_path(path: Path, resources_dir: Path) -> str:
    """Canonicalize a resource-relative path for collision-resistant FNV-1a hashing.

    Transforms applied in order:
    (1) POSIX relative path, lowercased.
    (2) Strip control bytes \x00-\x1f and \x7f — removes null/newline injection
        hazards.  After this step, \x01 (the RLE sentinel) cannot appear in the
        string, making it safe to use as an unambiguous marker in step (5).
    (3) Normalize backslash to forward slash.
    (4) Component-wise path resolution (split on '/'):
          a. Strip trailing spaces first — catches the NTFS ".." bypass where
             ".. " (dot-dot-space) names the parent directory on Windows.
          b. Resolve ".." by popping the last resolved component (path traversal).
          c. Skip "." and empty components (handles "//", "./").
          d. Strip trailing dots — on NTFS, "file." and "file" name the same
             resource; normalizing prevents silent collision across platforms.
          e. Skip components that became empty after dot-stripping (e.g. "...").
          f. Append the cleaned component to the resolved list.
    (5) Generalized RLE with \x01 sentinel: any run of N >= 2 equal characters
        encodes to \x01 <char> <N:02x> (fixed 2-hex-digit count, lowercase).
        Runs > 255 are split into chained tokens; single chars pass through.
        This extends the prior '0'-only encoding to all characters, adding
        diffusion wherever underscores, dashes, digits, or letters repeat.

    spectral_resource_path_canonical() in spectral_resource_fs.c applies the
    identical transforms.  Both sides must be kept in byte-exact sync or
    embedded ID lookups will silently fail.
    """
    # (1)
    s = path.relative_to(resources_dir).as_posix().lower()
    # (2) strip control characters — makes \x01 safe as sentinel
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    # (3)
    s = s.replace('\\', '/')

    # (4) component-wise path resolution
    resolved: list[str] = []
    for comp in s.split('/'):
        comp = comp.rstrip(' ')              # (4a) spaces before ".." check
        if comp == '..':
            if resolved:
                resolved.pop()             # (4b) path traversal
        elif comp in ('', '.'):
            pass                           # (4c) empty / current-dir
        else:
            comp = comp.rstrip('.')        # (4d) NTFS trailing-dot quirk
            if comp:
                resolved.append(comp)      # (4f) or skip if now empty (4e)

    s = '/'.join(resolved)

    # (5) generalized RLE with fixed-width 4-byte tokens: \x01 <char> <hi> <lo>
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        run = 1
        while i + run < len(s) and s[i + run] == c:
            run += 1
        remaining = run
        while remaining > 0:
            chunk = min(remaining, 255)
            if chunk >= 2:
                out.append(f'{_RLE_SENTINEL}{c}{chunk:02x}')
            else:
                out.append(c)
            remaining -= chunk
        i += run

    return ''.join(out)


def hash_file_fnv1a(path: Path) -> int:
    """Compute FNV-1a (32-bit) hash of a file's raw byte content.

    No external dependency — useful when xxhsum/xxhash is unavailable at
    table-generation time.  The same algorithm as file_id_from_path(), but
    applied to file bytes rather than path strings, so the two results are
    semantically distinct and never confused.
    """
    h = _FNV1A_32_OFFSET_BASIS
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            for byte in chunk:
                h ^= byte
                h = (h * _FNV1A_32_PRIME) & _FNV1A_32_MASK
    return h


def hash_file_xxhash(path: Path, xxhsum_bin: str, algorithm: str) -> int:
    if xxhash is not None:
        if algorithm == "xxh3_64":
            hasher = xxhash.xxh3_64()
        elif algorithm == "xxh32":
            hasher = xxhash.xxh32()
        else:
            raise ValueError(f"unsupported algorithm: {algorithm}")

        with path.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)

        return hasher.intdigest()

    if algorithm == "xxh3_64":
        commands = [
            [xxhsum_bin, "--xxh3", str(path)],
            [xxhsum_bin, "-H3", str(path)],
        ]
    elif algorithm == "xxh32":
        commands = [[xxhsum_bin, "-H0", str(path)]]
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    result = None
    last_error = None
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            break
        except subprocess.CalledProcessError as error:
            last_error = error

    if result is None:
        raise RuntimeError(
            f"xxhsum failed for {path} using {algorithm}: {last_error}"
        )

    # Output format: "<hex>  <path>"
    digest_hex = result.stdout.strip().split()[0]
    return int(digest_hex, 16)


def c_escape(value: str) -> str:
    """Escape a string for use as a C string literal.

    Handles all characters that would otherwise produce invalid or ambiguous C:
    backslash, double-quote, and the standard single-character escapes (\\t, \\n,
    \\r).  Any remaining non-printable or non-ASCII byte is emitted as \\xNN so
    the literal is always valid ASCII source.  Printable ASCII (0x20–0x7e,
    excluding \\ and ") passes through unchanged.
    """
    out = []
    for ch in value:
        if ch == '\\':
            out.append('\\\\')
        elif ch == '"':
            out.append('\\"')
        elif ch == '\t':
            out.append('\\t')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\r':
            out.append('\\r')
        elif 0x20 <= ord(ch) <= 0x7e:
            out.append(ch)
        else:
            out.append(f'\\x{ord(ch):02x}')
    return ''.join(out)


def find_repo_root(path: Path) -> Path | None:
    cursor = path
    while True:
        if (cursor / ".git").exists():
            return cursor
        if cursor.parent == cursor:
            return None
        cursor = cursor.parent


def list_resource_files(resources_dir: Path) -> list[Path]:
    repo_root = find_repo_root(resources_dir)
    git_bin = shutil.which("git")

    if repo_root is not None and git_bin is not None:
        rel_resources = resources_dir.relative_to(repo_root).as_posix()
        # Use only --cached (tracked files) for deterministic output.
        # --others would include untracked files whose presence varies by workspace.
        command = [
            git_bin,
            "-C",
            str(repo_root),
            "ls-files",
            "--cached",
            "--",
            rel_resources,
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        files: list[Path] = []
        for line in result.stdout.splitlines():
            if not line:
                continue
            candidate = (repo_root / line).resolve()
            if not candidate.is_file():
                continue
            try:
                candidate.relative_to(resources_dir)
            except ValueError:
                continue
            files.append(candidate)
        files.sort(key=lambda path: path.relative_to(resources_dir).as_posix())
        return files

    files = [path for path in resources_dir.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.relative_to(resources_dir).as_posix())
    return files


def collect_entries(
    resources_dir: Path,
    xxhsum_bin: str,
    embedded_algo: str = "xxh32",
) -> list[ResourceHashEntry]:
    files = list_resource_files(resources_dir)

    entries: list[ResourceHashEntry] = []
    for path in files:
        rel = path.relative_to(resources_dir).as_posix()
        canonical = compress_path(path, resources_dir)
        host_hash = hash_file_xxhash(path, xxhsum_bin, "xxh3_64")
        if embedded_algo == "fnv1a":
            embedded_hash = hash_file_fnv1a(path)
        else:  # xxh32
            embedded_hash = hash_file_xxhash(path, xxhsum_bin, "xxh32")
        file_size = path.stat().st_size
        entries.append(
            ResourceHashEntry(
                relative_path=rel,
                file_id=file_id_from_path(canonical),
                host_digest=host_hash,
                embedded_digest=embedded_hash,
                size_bytes=file_size,
            )
        )
    return entries


def generate_c(entries: list[ResourceHashEntry], embedded_algo: str = "xxh32") -> str:
    lines: list[str] = []
    lines.append("/* AUTO-GENERATED by tools/gen_resource_hashes.py — do not edit */")
    lines.append(f"/* Hash algorithms: host=xxh3_64, embedded={embedded_algo} */")
    lines.append("/*")
    lines.append(" * Note:")
    lines.append(" *   Preprocessor-time file hashing is not supported by C/C11.")
    lines.append(" *   Build-time equivalence is enforced by CMake verify_resource_hashes target.")
    lines.append(" *")
    lines.append(" * Example (commented) static assert style:")
    lines.append(" *   // _Static_assert(1, \"resource hash equivalence verified in build step\");")
    lines.append(" */")
    lines.append("#include <stddef.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include \"spectral_resource_fs.h\"")
    lines.append("")
    lines.append("const SpectralResourceFsEntry spectral_resource_hashes[] = {")
    lines.append("#if SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM")
    for entry in entries:
        lines.append(
            f'    {{ UINT32_C(0x{entry.file_id:08x}), UINT32_C(0x{entry.embedded_digest:08x}), (size_t){entry.size_bytes}u }},'
        )
    lines.append("#else")
    for entry in entries:
        lines.append(
            f'    {{"{c_escape(entry.relative_path)}", UINT64_C(0x{entry.host_digest:016x}), (size_t){entry.size_bytes}u}},'
        )
    lines.append("#endif")

    lines.append("};")
    lines.append("")
    lines.append(f"const size_t spectral_resource_hashes_count = (size_t){len(entries)}u;")
    lines.append("")
    return "\n".join(lines)


def _read_utf8_lf(path: Path) -> str:
    """Read a text file normalizing to LF line endings regardless of platform."""
    with path.open("rb") as f:
        raw = f.read()
    return raw.replace(b"\r\n", b"\n").decode("utf-8")


def _write_utf8_lf(path: Path, content: str) -> None:
    """Write content with LF line endings regardless of platform."""
    with path.open("wb") as f:
        f.write(content.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C resource hash array")
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=Path("resources"),
        help="Directory to scan recursively (default: resources)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spectral_engine/core/spectral_hash_resources_xx32_xx3.c"),
        help="Output C source path",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "verify"],
        default="generate",
        help="generate: write output, verify: compare output against fresh generation",
    )
    parser.add_argument(
        "--xxhsum-bin",
        type=str,
        default="xxhsum",
        help="xxhsum executable to use for hashing",
    )
    parser.add_argument(
        "--hash-embedded-algo",
        choices=["xxh32", "fnv1a"],
        default="xxh32",
        help=(
            "Content hash algorithm for the embedded digest field. "
            "xxh32 (default): stronger, requires xxhsum or Python xxhash module. "
            "fnv1a: no external dependency, always available, same algorithm as "
            "file_id but applied to file content rather than the path."
        ),
    )
    args = parser.parse_args()

    resources_dir = args.resources_dir.resolve()
    if not resources_dir.exists() or not resources_dir.is_dir():
        raise SystemExit(f"resources directory not found: {resources_dir}")

    xxhsum_bin = shutil.which(args.xxhsum_bin)
    if xxhash is None and not xxhsum_bin:
        raise SystemExit(
            "No xxHash backend available. Install either:\n"
            "  - Python module: pip install xxhash\n"
            "  - xxhsum CLI and pass --xxhsum-bin if needed"
        )

    if xxhash is not None:
        print(f"Using Python xxhash module (xxh3_64 host, {args.hash_embedded_algo} embedded)")
    else:
        print(f"Using xxhsum CLI ({xxhsum_bin}) for host; {args.hash_embedded_algo} for embedded")

    entries = collect_entries(resources_dir, xxhsum_bin, args.hash_embedded_algo)
    content = generate_c(entries, args.hash_embedded_algo)

    if args.mode == "generate":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_utf8_lf(args.output, content)
        print(
            f"Generated {args.output} with {len(entries)} entries from {resources_dir}"
        )
        return

    if not args.output.exists():
        raise SystemExit(f"verify failed: output file does not exist: {args.output}")

    existing = _read_utf8_lf(args.output)
    if existing != content:
        raise SystemExit(
            "verify failed: generated hashes do not match committed/generated output. "
            "Run in generate mode to refresh."
        )

    print(f"Verified {args.output}: hashes match {len(entries)} resources")


if __name__ == "__main__":
    main()
