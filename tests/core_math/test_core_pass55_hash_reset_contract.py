#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_hash_reset_validates_descriptor_before_reset_impl() -> None:
    src = read("spectral_engine/core/spectral_hash_xx32_xx3.c")

    assert "SpectralError spectral_hash_file_method_reset(SpectralHashFileMethod* method)" in src
    assert "const SpectralHashFileMethodDescriptor* desc = NULL;" in src
    assert "spectral_hash_file_method_get_descriptor(method->type, &desc)" in src
    assert "return spectral_hash_method_reset_impl(method);" in src

    descriptor_idx = src.index("spectral_hash_file_method_get_descriptor(method->type, &desc)")
    impl_idx = src.index("return spectral_hash_method_reset_impl(method);", descriptor_idx)
    assert descriptor_idx < impl_idx

def test_pass55_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS55.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "hash file reset lifecycle contract" in notes
    assert "reset-before-init" in notes
    assert "pass 55 hash reset" in audit
    assert "for pass_num in range(1, 56):" in audit
