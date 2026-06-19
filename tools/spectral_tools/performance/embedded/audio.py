"""Render the QEMU fixture to a listenable WAV.

The counts rig (``qemu_main.c``) stays counts-only; this drives a SEPARATE harness
(``qemu_render_main.c``) that runs the SAME real ``spectral_arm32_process`` under QEMU and dumps
the rendered Q15 audio to a host ``.wav`` over semihosting -- so you can actually hear what the M7
binary produces. Q15 is 16-bit PCM, written through the freestanding ``spectral_wav.h`` header.
It also returns the rig's rotl5/xor CHECKSUM so a caller can confirm the dumped audio is
bit-identical to the measured counts render (same fixture -> same CHECKSUM).

CLI::

    python -m spectral_tools.performance.embedded.audio [out.wav]   # default: ./qemu_render.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

from ...core.process import run
from .counts import CHECKSUM_RE, RESULT_RE, _build_runner_elf
from .fixture import WorkloadFixture, default_fixture
from .toolchain import NATIVE_DIR, Toolchain, discover

RENDER_SRC = NATIVE_DIR / "qemu" / "qemu_render_main.c"
# The harness opens this fixed name relative to qemu's cwd (SPECTRAL_QEMU_RENDER_WAV).
_HARNESS_WAV = "qemu_render.wav"


def render_wav(tc: Toolchain, out_dir: Path, fixture: WorkloadFixture | None = None,
               *, wav_name: str = _HARNESS_WAV) -> tuple[Path, str | None]:
    """Build + run the render harness; return (wav_path, checksum).

    The harness writes the WAV relative to qemu's working directory, so we run with
    ``cwd=out_dir`` and the file lands in ``out_dir``; it is renamed to ``wav_name`` if needed.
    """
    if tc.qemu_system_arm is None:
        raise RuntimeError("qemu-system-arm unavailable; run toolchain.discover(need={'qemu'})")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = fixture or default_fixture()
    spec.validate()

    elf = _build_runner_elf(tc, spec, out_dir, main_src=RENDER_SRC)
    result = run(
        [str(tc.qemu_system_arm), "-M", "mps2-an500", "-semihosting",
         "-display", "none", "-serial", "none", "-monitor", "none",
         "-kernel", str(elf)],
        cwd=out_dir, timeout_sec=300, check=False,
    )
    stdout = (result.stdout + "\n" + result.stderr).replace("\r", "")
    verdict = RESULT_RE.search(stdout)
    if result.returncode != 0 or verdict is None or verdict.group(1) != "PASS":
        raise RuntimeError(f"qemu render failed (rc={result.returncode}):\n{stdout}")

    produced = out_dir / _HARNESS_WAV
    if not produced.exists():
        raise RuntimeError(f"render exited PASS but wrote no {produced}\n{stdout}")
    target = out_dir / wav_name
    if target != produced:
        produced.replace(target)
    checksum = CHECKSUM_RE.search(stdout)
    return target, (checksum.group(1) if checksum else None)


def main(argv: list[str]) -> int:
    out = Path(argv[1]).resolve() if len(argv) > 1 else Path(_HARNESS_WAV).resolve()
    repo_root = Path(__file__).resolve().parents[4]
    tc = discover(repo_root, need=frozenset({"qemu"}))
    wav, checksum = render_wav(tc, out.parent, wav_name=out.name)
    print(f"wrote {wav}  (frames from fixture, mono 16-bit PCM; render checksum={checksum})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
