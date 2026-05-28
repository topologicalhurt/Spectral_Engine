# Patch notes — Pass 136: ARM32 DMA placement/coherency contract

## Problem

The DMA scratch buffer was hardwired to DTCM and the DMA completion path only used a barrier. On Cortex-M7-class systems, DMA accessibility and cacheability are SoC-specific; a barrier alone is not a cache invalidation policy.

## Change

Make DMA scratch placement and cacheability explicit build contracts, default the DMA scratch buffer away from forced DTCM placement, add a centralized receive-sync helper, and invalidate D-cache for explicitly cacheable DMA buffers when CMSIS cache maintenance is available.

## Why minimal

The existing DMA path remains optional. This pass only prevents hidden memory-placement/coherency assumptions from masquerading as a portable ARM32 contract.
