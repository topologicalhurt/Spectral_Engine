#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_peak_model_setters_route_through_one_mutation_helper() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "spectral_tracker_update_peak_model_field" in src
    assert "SPECTRAL_TRACKER_MODEL_FIELD_WINDOW" in src
    assert "SPECTRAL_TRACKER_MODEL_FIELD_ESTIMATOR" in src
    assert "SPECTRAL_TRACKER_MODEL_FIELD_PHASE_POLICY" in src
    assert "SPECTRAL_TRACKER_MODEL_FIELD_AMPLITUDE_POLICY" in src
    assert src.count("spectral_tracker_update_peak_model_field(tracker,") == 4

def test_tracker_setters_no_longer_duplicate_current_model_loads() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")
    setters_region = src[src.index("void spectral_tracker_set_window_descriptor"):src.index("float spectral_tracker_get_threshsq")]

    assert "model = spectral_tracker_current_peak_model(tracker);" not in setters_region
    assert setters_region.count("(void)spectral_tracker_update_peak_model_field") == 4

def test_pass95_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS95.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker peak-model mutation consolidation" in notes
    assert "coupled contract" in notes
    assert "pass 95 tracker peak model" in audit
    assert "for pass_num in range(1, 96):" in audit
