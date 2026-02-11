/* spectral_macros.h - Compiler Hints and Utility Macros */
#ifndef SPECTRAL_MACROS_H
#define SPECTRAL_MACROS_H

/*
 * Stringify Macros (for embedding defines into shader source strings)
 */
#define SPECTRAL_XSTR(x) #x
#define SPECTRAL_STR(x)  SPECTRAL_XSTR(x)

/*
 * Loop Unrolling Hints
 */
#if defined(__GNUC__) && !defined(__clang__)
#define SPECTRAL_UNROLL_4       _Pragma("GCC unroll 4")
#define SPECTRAL_UNROLL_2       _Pragma("GCC unroll 2")
#elif defined(__clang__)
#define SPECTRAL_UNROLL_4       _Pragma("clang loop unroll_count(4)")
#define SPECTRAL_UNROLL_2       _Pragma("clang loop unroll_count(2)")
#else
#define SPECTRAL_UNROLL_4
#define SPECTRAL_UNROLL_2
#endif

/*
 * Branch Prediction Hints
 */
#if defined(__GNUC__) || defined(__clang__)
#define SPECTRAL_LIKELY(x)      __builtin_expect(!!(x), 1)
#define SPECTRAL_UNLIKELY(x)    __builtin_expect(!!(x), 0)
#else
#define SPECTRAL_LIKELY(x)      (x)
#define SPECTRAL_UNLIKELY(x)    (x)
#endif

/*
 * Utility Macros
 */
#ifndef MAX
#define MAX(a, b)               (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b)               (((a) < (b)) ? (a) : (b))
#endif
#ifndef CLAMP
#define CLAMP(x, lo, hi)        (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))
#endif

/*
 * Memory Prefetch Hints
 */
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH_READ(addr)     __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr)    __builtin_prefetch((addr), 1, 3)
#else
#define PREFETCH_READ(addr)     ((void)0)
#define PREFETCH_WRITE(addr)    ((void)0)
#endif

/*
 * Force Inline Hint
 */
#ifndef SPECTRAL_FORCEINLINE
  #if defined(__GNUC__) || defined(__clang__)
    #define SPECTRAL_FORCEINLINE __attribute__((always_inline)) inline
  #elif defined(_MSC_VER)
    #define SPECTRAL_FORCEINLINE __forceinline
  #else
    #define SPECTRAL_FORCEINLINE inline
  #endif
#endif

#endif /* SPECTRAL_MACROS_H */
