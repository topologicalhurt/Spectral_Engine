#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_create_uses_checked_scalar_derivation() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "#include <float.h>" in src
    assert "spectral_tracker_derive_create_scalars" in src
    assert "n_freqs != ((size_t)n_fft / 2u + 1u)" in src
    assert "thresh_linear_sq_d = pow(10.0, (double)db_thresh / 10.0);" in src
    assert "threshsq_d = thresh_linear_sq_d * (double)max_magsq;" in src
    assert "freq_step_omega_d" in src
    assert "float thresh_linear = powf" not in src

def test_tracker_threshold_update_checks_new_max_and_product() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "double threshsq_d = 0.0;" in src
    assert "!isfinite(new_max_magsq) || new_max_magsq < 0.0f" in src
    assert "threshsq_d > (double)FLT_MAX" in src
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);" in src

def test_pass71_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS71.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker scalar-domain derivation contract" in notes
    assert "threshold" in notes
    assert "pass 71 tracker scalar" in audit
    assert "for pass_num in range(1, 72):" in audit
