#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_peak_phase_advance_uses_double_products_before_float_wrap() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "#include <float.h>" in src
    assert "double expected_center_d = 0.0;" in src
    assert "expected_center_d = (double)input->bin *" in src
    assert "residual_arg_d = (double)phase_delta - expected_center_d;" in src
    assert "denom_d = (double)input->freq_step_omega * (double)input->hop_float;" in src
    assert "fabs(residual_arg_d) > (double)FLT_MAX" in src
    assert "spectral_peak_wrap_phase_pi((float)residual_arg_d)" in src

def test_peak_phase_advance_checks_phase_omega_and_error_before_output() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "phase_omega_d = ((double)input->bin + (double)phase_bin_offset)" in src
    assert "phase_error_arg_d = (double)phase_delta -" in src
    assert "fabs(phase_omega_d) > (double)FLT_MAX" in src
    assert "fabs(phase_error_arg_d) > (double)FLT_MAX" in src

def test_pass75_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS75.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "peak phase-advance finite-product contract" in notes
    assert "phase" in notes
    assert "pass 75 peak phase advance" in audit
    assert "for pass_num in range(1, 76):" in audit
