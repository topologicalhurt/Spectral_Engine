"""Unit coverage for the benchmark summary parser and the runner's
warm/steady-state aggregation helpers.

These pin three correctness properties of the desktop performance harness:

  1. The last token on every summary line is parsed as a number, not a string
     that overran into the following line. The summary regexes capture with
     ``\\S+`` (not ``[^ ]+``); ``[^ ]`` matches newlines, so the trailing group
     used to swallow ``norm=1.6\\nStage...`` and fail the float parse. The bug
     was latent only because nothing downstream consumed ``norm_ms`` /
     ``warm_mean_ms`` — these tests fail loudly if it regresses.
  2. The per-stage breakdown is reported over the WARM set (cold first run
     discarded), reconciling with the headline ``warm_median`` instead of
     silently folding the cold outlier into the stage medians.
  3. The warm spread (min..max) is parsed, so a median's run-to-run jitter is
     legible rather than hidden behind a single number.

No binary or cross-compiler toolchain is needed: the parser operates on text.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from spectral_tools.testing.benchmark_parsing import BenchmarkOutputParser  # noqa: E402
from spectral_tools.testing.benchmark_runner import BenchmarkRunner  # noqa: E402


# A representative normal-mode summary block as emitted by BenchmarkRunner.run().
WARM_SUMMARY = (
    "Total ms: first=106.300 median=76.550 mean=80.783 warm_median=71.700 warm_mean=75.680\n"
    "Stage warm medians ms: fft=40.500 track=11.500 synth_kernel=7.900 synth_wall=8.518 norm=1.600\n"
    "Stage warm spread ms (min..max): fft=38.80..44.30 track=10.60..12.30 "
    "synth_kernel=6.70..15.70 synth_wall=7.45..16.28 norm=1.10..3.20\n"
    "Stage bandwidth warm medians: fft=25.330GiB/s (53.612% of faster stage) "
    "track=46.500GiB/s (100.000% of faster stage)\n"
    "Bandwidth percentages are relative between FFT and track (not hardware peak utilization).\n"
    "Memory max RSS (MB): median=256.218 mean=256.234 warm_median=256.203 warm_mean=256.225\n"
)

# The legacy (pre-warm-label, cold-inclusive) format must still parse.
LEGACY_SUMMARY = (
    "Total ms: first=106.300 median=76.550 mean=80.783 warm_median=71.700 warm_mean=75.680\n"
    "Stage medians ms: fft=40.500 track=11.500 synth_kernel=7.900 synth_wall=8.518 norm=1.600\n"
    "Stage bandwidth medians: fft=25.330GiB/s (53.612% of faster stage) "
    "track=46.500GiB/s (100.000% of faster stage)\n"
    "Memory max RSS (MB): median=256.218 mean=256.234 warm_median=256.203 warm_mean=256.225\n"
)


def test_last_token_on_each_line_is_a_number_not_an_overrun():
    """Regression: norm_ms and warm_mean_ms are the final group on their line;
    they must not swallow the following line."""
    m = BenchmarkOutputParser().parse_benchmark_output(WARM_SUMMARY)
    assert m["stage_medians"]["norm_ms"] == 1.6
    assert m["summary"]["warm_mean_ms"] == 75.680
    assert m["memory"]["rss_warm_mean_mb"] == 256.225


def test_warm_stage_medians_parse_with_full_field_set():
    m = BenchmarkOutputParser().parse_benchmark_output(WARM_SUMMARY)
    stage = m["stage_medians"]
    assert stage["fft_ms"] == 40.5
    assert stage["track_ms"] == 11.5
    assert stage["synth_kernel_ms"] == 7.9
    assert stage["synth_wall_ms"] == 8.518
    # synth_ms mirrors synth_kernel_ms for back-compat consumers.
    assert stage["synth_ms"] == 7.9


def test_warm_spread_band_is_parsed_per_stage():
    m = BenchmarkOutputParser().parse_benchmark_output(WARM_SUMMARY)
    spread = m["stage_spread"]
    assert spread["synth_kernel_min_ms"] == 6.70
    assert spread["synth_kernel_max_ms"] == 15.70
    assert spread["fft_min_ms"] == 38.80
    assert spread["norm_max_ms"] == 3.20


def test_bandwidth_warm_medians_parse():
    m = BenchmarkOutputParser().parse_benchmark_output(WARM_SUMMARY)
    bw = m["bandwidth_medians"]
    assert bw["fft_gibps"] == 25.330
    assert bw["track_rel_pct"] == 100.0


def test_legacy_cold_inclusive_labels_still_parse():
    """Archived reports used the unlabelled 'Stage medians ms:' /
    'Stage bandwidth medians:' lines. The optional '(?:warm )?' keeps them
    readable so old JSON does not silently lose its stage breakdown."""
    m = BenchmarkOutputParser().parse_benchmark_output(LEGACY_SUMMARY)
    assert m["stage_medians"]["norm_ms"] == 1.6
    assert m["bandwidth_medians"]["fft_gibps"] == 25.330
    # No spread line in the legacy format → empty section, not an error.
    assert m["stage_spread"] == {}


def test_runner_warm_drops_only_the_cold_first_run():
    assert BenchmarkRunner._warm([10.0, 2.0, 3.0, 4.0]) == [2.0, 3.0, 4.0]
    # A single run has nothing to warm — return it unchanged, never empty.
    assert BenchmarkRunner._warm([5.0]) == [5.0]
    assert BenchmarkRunner._warm([]) == []


def test_runner_spread_formats_band_and_handles_empty():
    assert BenchmarkRunner._fmt_spread([6.7, 15.7, 7.9]) == "6.70..15.70"
    assert BenchmarkRunner._fmt_spread([]) == "nan..nan"
