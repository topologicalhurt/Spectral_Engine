/* Phase 4 vDSP math-acceleration audit bench (host/Apple only; NOT wired into CMake).
 *
 * Reproduction harness for docs/core_audit/archive/VDSP_MATH_ACCEL_AUDIT.md
 * (the completed audit). Intentionally has NO CMake target — it links the Accelerate
 * framework, which the production host build does not, and is Apple-exclusive. By
 * design, not limbo: this is the documented manual reproduction recipe. Build + run:
 *
 *   clang -O3 -ffast-math -ffp-contract=fast -march=native -std=c11 \
 *     -Ispectral_engine/core -Ispectral_engine/arch/simd -Ithird_party/simde \
 *     tests/core_contracts/bench_vdsp_audit.c \
 *     spectral_engine/arch/simd/spectral_vector_ops.c \
 *     -framework Accelerate -lm -o /tmp/bench_vdsp_audit && /tmp/bench_vdsp_audit
 *
 * Times each production SIMDe vector op (the REAL spectral_vector_ops.c, compiled
 * with the production host flags -O3 -ffast-math -ffp-contract=fast -march=native)
 * against its vDSP / vForce equivalent, on representative sizes. Reports ns/elem,
 * speedup (SIMDe_time / vDSP_time; >1 means vDSP faster), and numeric divergence
 * (max-abs + RMS) so any bit-identity loss is explicit. NO production code is wired;
 * this only measures, per the measure-first / decline-on-data rule. */
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "spectral_vector_ops.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float frand(void) { return (float)((double)rand() / RAND_MAX * 2.0 - 1.0); }

static void diff_stats(const float* a, const float* b, size_t n, double* max_abs, double* rms) {
    double mx = 0.0, ss = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        double ad = fabs(d);
        if (ad > mx) mx = ad;
        ss += d * d;
    }
    *max_abs = mx;
    *rms = (n > 0) ? sqrt(ss / (double)n) : 0.0;
}

/* pick an iteration count so a single timed run is ~order 100ms regardless of size */
static long iters_for(size_t n) {
    long target = (long)(3.0e8 / (double)n);
    if (target < 50) target = 50;
    if (target > 2000000) target = 2000000;
    return target;
}

#define HDR() printf("%-16s %8s %10s %10s %9s  %-11s %-11s\n", \
    "op", "n", "SIMDe ns/e", "vDSP ns/e", "speedup", "max|diff|", "rms diff")

static void row(const char* op, size_t n, double simde_total, double vdsp_total,
                long it, double max_abs, double rms) {
    double simde_ns = simde_total / (double)(it) / (double)n * 1e9;
    double vdsp_ns  = vdsp_total  / (double)(it) / (double)n * 1e9;
    printf("%-16s %8zu %10.3f %10.3f %8.2fx  %-11.3e %-11.3e\n",
           op, n, simde_ns, vdsp_ns, simde_ns / vdsp_ns, max_abs, rms);
}

int main(void) {
    const size_t sizes[] = { 256, 513, 2049, 4097, 65536 };
    const size_t NS = sizeof(sizes) / sizeof(sizes[0]);
    const size_t NMAX = 65536;

    float *a   = malloc(NMAX * sizeof(float));
    float *b   = malloc(NMAX * sizeof(float));
    float *d1  = malloc(NMAX * sizeof(float));
    float *d2  = malloc(NMAX * sizeof(float));
    float *re  = malloc(NMAX * sizeof(float));
    float *im  = malloc(NMAX * sizeof(float));
    float *inter = malloc(2 * NMAX * sizeof(float));
    float *ph1 = malloc(NMAX * sizeof(float));
    float *ph2 = malloc(NMAX * sizeof(float));
    float *split_re = malloc(NMAX * sizeof(float));
    float *split_im = malloc(NMAX * sizeof(float));

    srand(12345);
    for (size_t i = 0; i < NMAX; i++) {
        a[i] = frand(); b[i] = frand();
        re[i] = frand(); im[i] = frand();
        inter[2*i] = re[i]; inter[2*i+1] = im[i];
    }

    double t0, t1, mx, rms; long it; volatile float sink = 0.0f;

    printf("\n=== Phase 4 vDSP audit (Apple, -O3 -ffast-math -march=native; SPECTRAL_ENABLE_APPROX_ATAN2=%d) ===\n\n",
           (int)SPECTRAL_ENABLE_APPROX_ATAN2);

    /* ---- vmul ---- */
    HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n);
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_vmul(a,b,d1,n); sink+=d1[0]; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vDSP_vmul(a,1,b,1,d2,1,n); sink+=d2[0]; } t1 = now_s(); double vt=t1-t0;
        diff_stats(d1,d2,n,&mx,&rms); row("vmul",n,st,vt,it,mx,rms);
    }
    /* ---- vadd ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n);
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_vadd(a,b,d1,n); sink+=d1[0]; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vDSP_vadd(a,1,b,1,d2,1,n); sink+=d2[0]; } t1 = now_s(); double vt=t1-t0;
        diff_stats(d1,d2,n,&mx,&rms); row("vadd",n,st,vt,it,mx,rms);
    }
    /* ---- vsq ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n);
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_vsq(a,d1,n); sink+=d1[0]; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vDSP_vsq(a,1,d2,1,n); sink+=d2[0]; } t1 = now_s(); double vt=t1-t0;
        diff_stats(d1,d2,n,&mx,&rms); row("vsq",n,st,vt,it,mx,rms);
    }
    /* ---- vsmul ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n); float sc = 0.37f;
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_vsmul(a,sc,d1,n); sink+=d1[0]; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vDSP_vsmul(a,1,&sc,d2,1,n); sink+=d2[0]; } t1 = now_s(); double vt=t1-t0;
        diff_stats(d1,d2,n,&mx,&rms); row("vsmul",n,st,vt,it,mx,rms);
    }
    /* ---- vmax (max value) ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n); float o1,o2;
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_vmax(a,&o1,n); sink+=o1; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vDSP_maxv(a,1,&o2,n); sink+=o2; } t1 = now_s(); double vt=t1-t0;
        mx = fabs((double)o1-(double)o2); rms = mx; row("vmax",n,st,vt,it,mx,rms);
    }
    /* ---- vmaxmgv (max magnitude) ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n); float o1,o2;
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_vmaxmgv(a,&o1,n); sink+=o1; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vDSP_maxmgv(a,1,&o2,n); sink+=o2; } t1 = now_s(); double vt=t1-t0;
        mx = fabs((double)o1-(double)o2); rms = mx; row("vmaxmgv",n,st,vt,it,mx,rms);
    }
    /* ---- atan2: production path (scalar atan2f since approx=0) vs vForce vvatan2f ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n); int ni=(int)n;
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_vatan2(a,b,d1,n); sink+=d1[0]; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vvatan2f(d2,a,b,&ni); sink+=d2[0]; } t1 = now_s(); double vt=t1-t0;
        diff_stats(d1,d2,n,&mx,&rms); row("atan2",n,st,vt,it,mx,rms);
    }
    /* ---- magsq_split: SIMDe vs vDSP_zvmags (split complex) ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n);
        DSPSplitComplex sc = { .realp = re, .imagp = im };
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_magsq_split(re,im,d1,n); sink+=d1[0]; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){ vDSP_zvmags(&sc,1,d2,1,n); sink+=d2[0]; } t1 = now_s(); double vt=t1-t0;
        diff_stats(d1,d2,n,&mx,&rms); row("magsq_split",n,st,vt,it,mx,rms);
    }
    /* ---- magsq_only (interleaved): SIMDe vs vDSP ctoz+zvmags+maxv ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n);
        float m1,m2; DSPSplitComplex sc = { .realp = split_re, .imagp = split_im };
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_magsq_only(inter,d1,&m1,n); sink+=m1; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){
            vDSP_ctoz((const DSPComplex*)inter,2,&sc,1,n);
            vDSP_zvmags(&sc,1,d2,1,n);
            vDSP_maxv(d2,1,&m2,n); sink+=m2;
        } t1 = now_s(); double vt=t1-t0;
        diff_stats(d1,d2,n,&mx,&rms); double dm=fabs((double)m1-(double)m2);
        row("magsq_only",n,st,vt,it, mx>dm?mx:dm, rms);
    }
    /* ---- magsq_phase (interleaved): SIMDe (scalar atan2f) vs vDSP ctoz+zvmags+maxv+vvatan2f ---- */
    printf("\n"); HDR();
    for (size_t s = 0; s < NS; s++) { size_t n = sizes[s]; it = iters_for(n); int ni=(int)n;
        float m1,m2; DSPSplitComplex sc = { .realp = split_re, .imagp = split_im };
        t0 = now_s(); for (long k=0;k<it;k++){ spectral_magsq_phase(inter,d1,ph1,&m1,n); sink+=ph1[0]; } t1 = now_s(); double st=t1-t0;
        t0 = now_s(); for (long k=0;k<it;k++){
            vDSP_ctoz((const DSPComplex*)inter,2,&sc,1,n);
            vDSP_zvmags(&sc,1,d2,1,n);
            vDSP_maxv(d2,1,&m2,n);
            vvatan2f(ph2,split_im,split_re,&ni); sink+=ph2[0];
        } t1 = now_s(); double vt=t1-t0;
        double mxm,rmm,mxp,rmp; diff_stats(d1,d2,n,&mxm,&rmm); diff_stats(ph1,ph2,n,&mxp,&rmp);
        row("magsq_phase",n,st,vt,it, mxp, rmp);  /* report phase divergence (the interesting one) */
    }

    printf("\n(magsq_phase row reports PHASE max|diff|/rms; magsq itself matches to ~%.0e)\n", 1e-6);
    if (sink == 12345.6789f) printf("");  /* keep sink live */
    free(a);free(b);free(d1);free(d2);free(re);free(im);free(inter);free(ph1);free(ph2);free(split_re);free(split_im);
    return 0;
}
