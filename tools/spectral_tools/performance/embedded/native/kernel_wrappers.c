/* Perf-model Layer 0/2 wrapper TU (M7_PERF_MODEL_PLAN P1).
 *
 * Includes the real ARM backend TU unmodified and exposes each static-inline
 * hot kernel behind an extern wrapper bracketed by llvm-mca region markers.
 * The kernels inline into the wrappers, so the marked regions contain the
 * exact -O3 loop bodies without touching kernel source. The volatile asm
 * markers sit outside the loops; they pin the region but do not constrain
 * scheduling inside it.
 *
 * Compiled by spectral_tools.performance.embedded.codegen.loop_analysis()
 * with the Daisy production flags (parsed from options.cmake at runtime).
 */

#include "spectral_synth_arm32.c"

#define MCA_BEGIN(name) __asm volatile("# LLVM-MCA-BEGIN " #name)
#define MCA_END()       __asm volatile("# LLVM-MCA-END")

void mca_synth_core_m7(q63_t* restrict accum, SpectralCoupledOsc* restrict osc,
                       q31_t cos_w, q31_t sin_w, uint32_t blk_start, uint32_t blk_end,
                       q15_t* restrict amp, q15_t amp_delta) {
    MCA_BEGIN(synth_core_m7);
    synth_core_m7(accum, osc, cos_w, sin_w, blk_start, blk_end, amp, amp_delta);
    MCA_END();
}

#if SPECTRAL_HAS_DUAL_MAC
void mca_synth_core_pair_m7(q63_t* restrict accum, uint32_t blk_start, uint32_t blk_end,
                            uint32_t* restrict phaseA, q15_t* restrict ampA,
                            uint32_t phase_incA, q15_t amp_deltaA,
                            uint32_t* restrict phaseB, q15_t* restrict ampB,
                            uint32_t phase_incB, q15_t amp_deltaB) {
    /* Rotation constants are per-voice activation work (computed once, not per block); the
     * production path stores them in the active record. Compute them here OUTSIDE the marked
     * region so the measured loop is the steady-state dual-MAC, matching the kernel. */
    q31_t cwA, swA, cwB, swB;
    spectral_coupled_freq_constants((double)phase_incA * SPECTRAL_PHASE_U32_TO_RAD, &cwA, &swA);
    spectral_coupled_freq_constants((double)phase_incB * SPECTRAL_PHASE_U32_TO_RAD, &cwB, &swB);
    MCA_BEGIN(synth_core_pair_m7);
    synth_core_pair_m7(accum, blk_start, blk_end,
                       phaseA, ampA, phase_incA, amp_deltaA, cwA, swA,
                       phaseB, ampB, phase_incB, amp_deltaB, cwB, swB);
    MCA_END();
}
#endif

void mca_synth_fade_m7(q63_t* restrict accum, SpectralCoupledOsc* restrict osc,
                       q31_t cos_w, q31_t sin_w, uint32_t blk_start, uint32_t blk_end,
                       q15_t* restrict amp, q15_t amp_delta,
                       q15_t fade_val, q15_t fade_step) {
    MCA_BEGIN(synth_fade_m7);
    synth_fade_m7(accum, osc, cos_w, sin_w, blk_start, blk_end,
                  amp, amp_delta, fade_val, fade_step);
    MCA_END();
}
