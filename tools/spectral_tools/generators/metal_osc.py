"""Generate or verify the Metal MSL oscillator shader from the C contract.

The Metal backend cannot include the C synthesis headers directly: its shader is
compiled at runtime from a string.  Historically the MSL was a hand-written
mirror of spectral_osc_formulas.h / spectral_segment_math.h, kept in sync only by
a `_Static_assert(VERSION == N)` reminder — a weak guard that fires only if a
formula edit *also* bumps the version (P4: silent Metal drift).

This tool makes the C headers the single source of truth.  It extracts the
drift-prone formula expressions (the algebraic waveforms + segment math), applies
the fixed C->MSL token policy (sin/asin -> native MSL intrinsics, floorf->floor,
fabsf->abs, SPECTRAL_<CONST> -> <CONST>), validates the stable GPU-policy bodies
(phase normalization, fade ramp) against the headers, and emits
core/spectral_osc_metal_generated.h.  CMake regenerates it every build; the
`verify` mode (mirroring resource_hashes.py) fails the build if the committed
header is not what the current C contract produces -> drift becomes impossible.

What stays POLICY (deliberate C<->MSL divergence, not extracted):
  - sine/asin use the GPU's NATIVE transcendentals (not the deg-9 minimax poly);
  - the oscillator switch covers only the 6 point-sampled timbres (quantized/pwm
    fold to the default sine case, exactly as the CPU GPU-tile path never feeds
    them a width);
  - constant *values* are still injected by the C preprocessor via SPECTRAL_STR
    so spectral_consts.h remains the one source for the numbers.

Usage (generate):
    PYTHONPATH=tools python3 -m spectral_tools.generators.metal_osc \
        --output spectral_engine/core/spectral_osc_metal_generated.h

Usage (verify):
    PYTHONPATH=tools python3 -m spectral_tools.generators.metal_osc \
        --output spectral_engine/core/spectral_osc_metal_generated.h --mode verify
"""

from __future__ import annotations

import argparse
import re
import sys

from pathlib import Path

# When executed directly (python metal_osc.py), __package__ is unset.
if __package__ in {None, ""}:
    _tools_root = str(Path(__file__).resolve().parents[2])
    if _tools_root not in sys.path:
        sys.path.insert(0, _tools_root)
    __package__ = "spectral_tools.generators"  # noqa: A001

from ..core.console import Console
from ..core.utils import read_utf8_lf, write_utf8_lf

# Repo-relative defaults (resolved against the repo root = tools/..).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE = _REPO_ROOT / "spectral_engine" / "core"
DEFAULT_FORMULAS = _CORE / "spectral_osc_formulas.h"
DEFAULT_SEGMENT_MATH = _CORE / "spectral_segment_math.h"
DEFAULT_OSCILLATOR_H = _CORE / "oscillator.h"
DEFAULT_OUTPUT = _CORE / "spectral_osc_metal_generated.h"
DEFAULT_MODE = "generate"
MODE_CHOICES = ("generate", "verify")

# C identifier -> MSL spelling for constants injected via the SPECTRAL_STR macros
# emitted in the shader's #define block.  Longest-first so INV_TWO_PI is matched
# before TWO_PI and INV_PI_SQ before INV_PI.
_CONST_MAP = [
    ("SPECTRAL_INV_TWO_PI", "INV_TWO_PI"),
    ("SPECTRAL_INV_PI_SQ", "INV_PI_SQ"),
    ("SPECTRAL_TWO_PI", "TWO_PI"),
    ("SPECTRAL_INV_PI", "INV_PI"),
    ("SPECTRAL_PI", "PI"),
]
# C intrinsic / helper -> MSL intrinsic.
_FUNC_MAP = [
    ("floorf", "floor"),
    ("fabsf", "abs"),
    ("spectral_fast_sin_inline", "oscillator_fast_sin"),
]

# Timbres the MSL switch renders with native transcendentals instead of the
# extracted algebraic expression.
_SINE_CASE = "oscillator_fast_sin(rads)"
_ASIN_CASE = "asin(clamp(rads * INV_PI, -1.0f, 1.0f))"
_POLICY_CASES = {
    "spectral_osc_sine": _SINE_CASE,
    "spectral_osc_asin": _ASIN_CASE,
}
# Timbres with width-dependent / multi-statement bodies the GPU tile path never
# exercises (it passes no width); they fold to the default sine case, matching
# the pre-codegen MSL.
_EXCLUDED_WAVEFORMS = {"spectral_osc_quantized", "spectral_osc_pwm"}


class GeneratorError(RuntimeError):
    """Raised when the C contract is not in the shape the generator expects.

    A loud failure here is the point: it means a header was reshaped in a way the
    Metal codegen no longer understands, so the build stops instead of emitting a
    silently-wrong shader.
    """


def _strip_block_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def rewrite_to_msl(expr: str) -> str:
    """Apply the fixed C->MSL token policy to a single extracted expression."""
    out = expr
    for c_name, msl_name in _FUNC_MAP:
        out = re.sub(rf"\b{re.escape(c_name)}\b", msl_name, out)
    for c_name, msl_name in _CONST_MAP:
        out = re.sub(rf"\b{re.escape(c_name)}\b", msl_name, out)
    return re.sub(r"\s+", " ", out).strip()


def _func_body(source: str, name: str) -> str:
    """Return the brace body of `float NAME(...) { ... }` (no nested braces)."""
    pattern = re.compile(
        rf"\bfloat\s+{re.escape(name)}\s*\([^)]*\)\s*\{{([^{{}}]*)\}}",
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        raise GeneratorError(f"could not locate function body for {name}()")
    return match.group(1)


def _return_expr(source: str, name: str) -> str:
    """Extract the single `return <expr>;` from a one-return function body."""
    body = _func_body(source, name)
    returns = re.findall(r"\breturn\s+(.*?);", body, flags=re.DOTALL)
    if len(returns) != 1:
        raise GeneratorError(
            f"expected exactly one return in {name}(), found {len(returns)}"
        )
    return returns[0].strip()


def _params(source: str, name: str) -> str:
    """Extract the parameter list text of `float NAME(<params>)`."""
    match = re.search(
        rf"\bfloat\s+{re.escape(name)}\s*\(([^)]*)\)", source, flags=re.DOTALL
    )
    if not match:
        raise GeneratorError(f"could not locate parameter list for {name}()")
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _version(source: str, macro: str) -> str:
    match = re.search(rf"#define\s+{re.escape(macro)}\s+(\d+)", source)
    if not match:
        raise GeneratorError(f"could not find {macro}")
    return match.group(1)


def parse_timbre_list(oscillator_h: str) -> list[tuple[str, str]]:
    """Parse SPECTRAL_OSC_TIMBRE_LIST -> ordered [(TIMBRE_NAME, waveform_fn)]."""
    macro = re.search(
        r"#define\s+SPECTRAL_OSC_TIMBRE_LIST\(X\)(.*?)(?:\n\n|\n[^\s\\])",
        oscillator_h,
        flags=re.DOTALL,
    )
    if not macro:
        raise GeneratorError("could not find SPECTRAL_OSC_TIMBRE_LIST")
    pairs = re.findall(r"X\(\s*(TIMBRE_\w+)\s*,\s*(spectral_osc_\w+)\s*\)", macro.group(1))
    if not pairs:
        raise GeneratorError("SPECTRAL_OSC_TIMBRE_LIST parsed to zero entries")
    return pairs


def _expect(actual: str, expected: str, what: str) -> None:
    if actual != expected:
        raise GeneratorError(
            f"{what}: C contract no longer matches the generator's MSL template.\n"
            f"  derived from header : {actual}\n"
            f"  generator template  : {expected}\n"
            "Update the template in metal_osc.py (and re-verify Metal output) if "
            "this divergence is intended."
        )


def build_oscillator_source(formulas: str, timbres: list[tuple[str, str]]) -> list[str]:
    """Assemble the `oscillator_metal_source` MSL string, one C line per list item.

    Returns a list of raw MSL text lines (without the C string-literal wrapping).
    """
    # Supported subset = list order minus the excluded (width-dependent) waveforms.
    supported = [(t, fn) for (t, fn) in timbres if fn not in _EXCLUDED_WAVEFORMS]

    # Validate the stable GPU-policy bodies against the headers so a formula edit
    # cannot silently bypass the codegen.
    # Inline the `norm` local into the return expression.  This is NOT cosmetic:
    # a named intermediate `float norm = p * INV_TWO_PI;` forces a round to f32
    # and so DEFEATS the FMA contraction the MSL compiler applies to the inlined
    # `floor(p * INV_TWO_PI + 0.5f)`.  The contracted-vs-rounded result differs by
    # ~1 ULP near half-integer boundaries, which flips the floor() occasionally and
    # perturbs the normalized phase on non-trivial inputs (invisible on a steady
    # tone, audible/byte-different on speech).  Emit the inlined form to match the
    # CPU/CUDA `spectral_normalize_phase` and the historical MSL.
    norm_expr = _return_expr(formulas, "spectral_normalize_phase")
    norm_body = _func_body(formulas, "spectral_normalize_phase")
    norm_local = re.search(r"\bfloat\s+norm\s*=\s*(.*?);", norm_body)
    if not norm_local:
        raise GeneratorError("spectral_normalize_phase: expected a `norm` local")
    norm_inlined = rewrite_to_msl(
        re.sub(r"\bnorm\b", f"({norm_local.group(1).strip()})", norm_expr)
    )
    _expect(
        norm_inlined,
        "p - TWO_PI * floor((p * INV_TWO_PI) + 0.5f)",
        "phase normalization",
    )

    fade_ramp = rewrite_to_msl(_return_expr(formulas, "spectral_fade_envelope_in"))
    _expect(
        fade_ramp,
        "0.5f * (1.0f + oscillator_fast_sin((j * inv_fade - 0.5f) * PI))",
        "fade-in ramp",
    )
    fade_ramp_out = fade_ramp.replace("j * inv_fade", "from_end * inv_fade")

    # Derived waveform switch cases.
    def case_body(fn: str) -> str:
        if fn in _POLICY_CASES:
            return _POLICY_CASES[fn]
        return rewrite_to_msl(_return_expr(formulas, fn))

    define_lines = [f"#define {t:<14} {i}" for i, (t, _fn) in enumerate(supported)]
    pad = max(len(t) for t, _ in supported) + 2
    case_lines = [
        f"        case {t + ':':<{pad}} return {case_body(fn)};" for (t, fn) in supported
    ]

    lines: list[str] = []
    lines += define_lines
    lines.append("")
    lines.append('#define TWO_PI " SPECTRAL_STR(SPECTRAL_TWO_PI) "')
    lines.append('#define INV_TWO_PI " SPECTRAL_STR(SPECTRAL_INV_TWO_PI) "')
    lines.append('#define INV_PI " SPECTRAL_STR(SPECTRAL_INV_PI) "')
    lines.append('#define INV_PI_SQ " SPECTRAL_STR(SPECTRAL_INV_PI_SQ) "')
    lines.append('#define PI " SPECTRAL_STR(SPECTRAL_PI) "')
    lines.append("")
    lines.append("/* Must match spectral_fast_sin_inline() in spectral_osc_formulas.h.")
    lines.append(" * GPU policy: native sin, NOT the CPU deg-9 minimax poly. */")
    lines.append("inline float oscillator_fast_sin(float x) {")
    lines.append("    return sin(x);")
    lines.append("}")
    lines.append("")
    lines.append("/* Must match spectral_normalize_phase() in spectral_osc_formulas.h.")
    lines.append(" * Inlined (no `norm` local) to preserve FMA contraction parity. */")
    lines.append("inline float oscillator_normalize_phase(float p) {")
    lines.append(f"    return {norm_inlined};")
    lines.append("}")
    lines.append("")
    lines.append('#define FADE_SAMPLES " SPECTRAL_STR(SPECTRAL_FADE_SAMPLES_DESKTOP) "')
    lines.append("")
    lines.append("/* Must match spectral_fade_envelope_gpu() in spectral_osc_formulas.h */")
    lines.append("inline float fade_envelope(float j, float seg_len) {")
    lines.append("    float fade_len = min(seg_len * 0.25f, (float)FADE_SAMPLES);")
    lines.append("    if (fade_len < 1.0f) fade_len = 1.0f;")
    lines.append("    float inv_fade = 1.0f / fade_len;")
    lines.append("    if (j < fade_len) {")
    lines.append(f"        return {fade_ramp};")
    lines.append("    }")
    lines.append("    float from_end = seg_len - 1.0f - j;")
    lines.append("    if (from_end < fade_len) {")
    lines.append(f"        return {fade_ramp_out};")
    lines.append("    }")
    lines.append("    return 1.0f;")
    lines.append("}")
    lines.append("")
    lines.append("/* Must match spectral_osc_* waveforms in spectral_osc_formulas.h */")
    lines.append("inline float oscillator(float phase, uint timbre) {")
    lines.append("    float rads = oscillator_normalize_phase(phase);")
    lines.append("    switch (timbre) {")
    lines += case_lines
    lines.append("        default:" + " " * (pad - len("default:") + 7) + "return oscillator_fast_sin(rads);")
    lines.append("    }")
    lines.append("}")
    return lines


def build_segment_math_source(segment_math: str) -> list[str]:
    """Assemble the segment-math MSL helpers from spectral_segment_math.h.

    The Metal tile kernel uses QUADRATIC phase (spectral_segment_phase_at_f32);
    the cubic helper is CPU-only and deliberately not mirrored. """
    # (C function, MSL function) — MSL drops the `spectral_` prefix and `_f32`.
    helpers = [
        ("spectral_segment_alpha_f32", "segment_alpha"),
        ("spectral_segment_beta_f32", "segment_beta"),
        ("spectral_segment_d_amp_f32", "segment_d_amp"),
        ("spectral_segment_phase_at_f32", "segment_phase_at"),
        ("spectral_segment_amp_at_f32", "segment_amp_at"),
    ]
    lines: list[str] = []
    lines.append("/* Generated from spectral_segment_math.h — keep the Metal tile")
    lines.append(" * kernel on the QUADRATIC phase helper (cubic is CPU-only). */")
    for c_name, msl_name in helpers:
        params = _params(segment_math, c_name)
        expr = rewrite_to_msl(_return_expr(segment_math, c_name))
        lines.append(f"inline float {msl_name}({params}) {{")
        lines.append(f"    return {expr};")
        lines.append("}")
    return lines


def _as_c_string_block(name: str, lines: list[str]) -> list[str]:
    """Wrap raw MSL lines as a C `const char* name = "...";` definition.

    Each MSL line becomes a "...\\n" literal.  Lines carrying a SPECTRAL_STR(...)
    fragment already contain their own closing/opening quotes, so they are emitted
    verbatim (the preprocessor stringizes the constant value at C compile time). """
    out = [f"const char* {name} ="]
    for line in lines:
        if "SPECTRAL_STR(" in line:
            # e.g.  #define TWO_PI " SPECTRAL_STR(SPECTRAL_TWO_PI) "
            out.append(f'"{line}\\n"')
        else:
            escaped = line.replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'"{escaped}\\n"')
    out[-1] = out[-1] + ";"
    return out


def generate_header(
    formulas: str,
    segment_math: str,
    oscillator_h: str,
) -> str:
    timbres = parse_timbre_list(oscillator_h)
    osc_lines = build_oscillator_source(formulas, timbres)
    seg_lines = build_segment_math_source(segment_math)
    osc_version = _version(formulas, "SPECTRAL_OSC_FORMULAS_VERSION")
    seg_version = _version(segment_math, "SPECTRAL_SEGMENT_MATH_VERSION")

    header = [
        "/* AUTO-GENERATED by tools/spectral_tools/generators/metal_osc.py — DO NOT EDIT.",
        " *",
        " * Single source of truth for the Metal MSL oscillator + segment math:",
        " *   core/spectral_osc_formulas.h   (waveforms, phase, fade)",
        " *   core/spectral_segment_math.h   (segment alpha/beta/d_amp/phase/amp)",
        " *   core/oscillator.h              (SPECTRAL_OSC_TIMBRE_LIST ordering)",
        " *",
        " * Regenerate: cmake --build build --target generate_metal_osc",
        " * Verify:     cmake --build build --target verify_metal_osc",
        " *",
        f" * osc_formulas_version={osc_version} segment_math_version={seg_version}",
        " */",
        "#ifndef SPECTRAL_OSC_METAL_GENERATED_H",
        "#define SPECTRAL_OSC_METAL_GENERATED_H",
        "",
        "#if defined(__APPLE__) && !defined(__CUDACC__)",
        "",
    ]
    body = _as_c_string_block("oscillator_metal_source", osc_lines)
    body.append("")
    body += _as_c_string_block("spectral_segment_math_metal_source", seg_lines)
    footer = [
        "",
        "#endif /* __APPLE__ && !__CUDACC__ */",
        "",
        "#endif /* SPECTRAL_OSC_METAL_GENERATED_H */",
        "",
    ]
    return "\n".join([*header, *body, *footer])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Metal MSL oscillator header")
    parser.add_argument("--formulas", type=Path, default=DEFAULT_FORMULAS)
    parser.add_argument("--segment-math", type=Path, default=DEFAULT_SEGMENT_MATH)
    parser.add_argument("--oscillator-h", type=Path, default=DEFAULT_OSCILLATOR_H)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-m", "--mode", choices=MODE_CHOICES, default=DEFAULT_MODE)
    args = parser.parse_args()
    console = Console()

    try:
        content = generate_header(
            read_utf8_lf(args.formulas),
            read_utf8_lf(args.segment_math),
            read_utf8_lf(args.oscillator_h),
        )
    except GeneratorError as exc:
        console.print_error(f"metal_osc generation failed: {exc}")
        return 1

    if args.mode == "generate":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_utf8_lf(args.output, content)
        print(f"Wrote {args.output}")
        return 0

    if not args.output.exists():
        console.print_error(f"verify failed: output file does not exist: {args.output}")
        return 1
    if read_utf8_lf(args.output) != content:
        console.print_error(
            "verify failed: generated Metal MSL header does not match the current C "
            "contract. Run in generate mode to refresh (cmake --target generate_metal_osc)."
        )
        return 1
    print(f"Verified {args.output}: Metal MSL matches the C contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
