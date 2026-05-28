#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[2]
src = (root / "spectral_engine/synth/backends/arm/spectral_synth_arm32.c").read_text()
assert "SPECTRAL_ARM32_DMA_BUFFER_DTCM" in src
assert "SPECTRAL_ARM32_DMA_BUFFER_CACHEABLE" in src
assert "SPECTRAL_ARM32_DMA_BUFFER_ATTR" in src
assert "spectral_arm32_dma_rx_sync" in src
assert "SCB_InvalidateDCache_by_Addr" in src
assert "dma_prefetch_coherent" in src
assert "static SpectralSegmentQ15 dma_seg_buf[SPECTRAL_DMA_BATCH] SPECTRAL_DTCM" not in src
