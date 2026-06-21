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

import shutil
import sys
import tempfile
from pathlib import Path

from ...core.process import run
from .counts import CHECKSUM_RE, RESULT_RE, _build_runner_elf
from .fixture import WorkloadFixture, default_fixture
from .toolchain import NATIVE_DIR, Toolchain, discover

RENDER_SRC = NATIVE_DIR / "qemu" / "qemu_render_main.c"
# The harness opens this fixed name relative to qemu's cwd (SPECTRAL_QEMU_RENDER_WAV).
_HARNESS_WAV = "qemu_render.wav"


def render_wav(tc: Toolchain, wav_path: Path, fixture: WorkloadFixture | None = None,
               *, build_dir: Path | None = None) -> tuple[Path, str | None]:
    """Build the render harness, run it under QEMU, and write ONLY the rendered WAV to wav_path.

    The harness compile (ELF, the generated fixture header, object files) happens in a throwaway
    build dir so it never clutters the WAV's directory. Pass build_dir to keep the artifacts (for
    debugging the ELF); omit it to build in an auto-cleaned temp dir. Returns (wav_path, checksum)
    where checksum is the rig's rotl5/xor of the render (bit-identity check vs the counts rig).
    """
    if tc.qemu_system_arm is None:
        raise RuntimeError("qemu-system-arm unavailable; run toolchain.discover(need={'qemu'})")
    wav_path = wav_path.resolve()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    spec = fixture or default_fixture()
    spec.validate()

    tmp_ctx = (tempfile.TemporaryDirectory(prefix="spectral-qemu-render-")
               if build_dir is None else None)
    build = Path(tmp_ctx.name) if tmp_ctx is not None else build_dir.resolve()
    try:
        build.mkdir(parents=True, exist_ok=True)
        elf = _build_runner_elf(tc, spec, build, main_src=RENDER_SRC)
        # The harness writes the WAV relative to qemu's cwd, so run inside the build dir.
        result = run(
            [str(tc.qemu_system_arm), "-M", "mps2-an500", "-semihosting",
             "-display", "none", "-serial", "none", "-monitor", "none",
             "-kernel", str(elf)],
            cwd=build, timeout_sec=300, check=False,
        )
        stdout = (result.stdout + "\n" + result.stderr).replace("\r", "")
        verdict = RESULT_RE.search(stdout)
        if result.returncode != 0 or verdict is None or verdict.group(1) != "PASS":
            raise RuntimeError(f"qemu render failed (rc={result.returncode}):\n{stdout}")
        produced = build / _HARNESS_WAV
        if not produced.exists():
            raise RuntimeError(f"render exited PASS but wrote no WAV in {build}\n{stdout}")
        shutil.move(str(produced), str(wav_path))   # only the WAV leaves the build dir
        checksum = CHECKSUM_RE.search(stdout)
        return wav_path, (checksum.group(1) if checksum else None)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main(argv: list[str]) -> int:
    out = Path(argv[1]).resolve() if len(argv) > 1 else Path(_HARNESS_WAV).resolve()
    repo_root = Path(__file__).resolve().parents[4]
    tc = discover(repo_root, need=frozenset({"qemu"}))
    wav, checksum = render_wav(tc, out)
    print(f"wrote {wav}  (mono 16-bit PCM; render checksum={checksum})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
