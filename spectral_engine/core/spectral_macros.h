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
#elif defined(__clang__)
#define SPECTRAL_UNROLL_4       _Pragma("clang loop unroll_count(4)")
#else
#define SPECTRAL_UNROLL_4
#endif

/*
 * Branch Prediction Hints
 */
#if defined(__GNUC__) || defined(__clang__)
#define SPECTRAL_UNLIKELY(x)    __builtin_expect(!!(x), 0)
#else
#define SPECTRAL_UNLIKELY(x)    (x)
#endif

/*
 * Utility Macros
 *
 * Kept as macros (not static inline) because they are type-generic across
 * int/size_t/float call sites; #ifndef-guarded since system headers and
 * vendored code define the same names with identical semantics. They
 * MULTI-EVALUATE their arguments: never pass expressions with side effects.
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
#define SPECTRAL_PREFETCH_READ_LOCALITY(addr, locality)  __builtin_prefetch((addr), 0, (locality))
#define SPECTRAL_PREFETCH_WRITE_LOCALITY(addr, locality) __builtin_prefetch((addr), 1, (locality))
#define SPECTRAL_PREFETCH_READ(addr)                     SPECTRAL_PREFETCH_READ_LOCALITY((addr), 3)
#define SPECTRAL_PREFETCH_WRITE(addr)                    SPECTRAL_PREFETCH_WRITE_LOCALITY((addr), 3)
#else
#define SPECTRAL_PREFETCH_READ_LOCALITY(addr, locality)  ((void)0)
#define SPECTRAL_PREFETCH_WRITE_LOCALITY(addr, locality) ((void)0)
#define SPECTRAL_PREFETCH_READ(addr)                     ((void)0)
#define SPECTRAL_PREFETCH_WRITE(addr)                    ((void)0)
#endif

/*
 * Force Inline Hint
 */
#ifndef SPECTRAL_FORCEINLINE
  #if SPECTRAL_EMBEDDED
    #if defined(__GNUC__) || defined(__clang__)
      #define SPECTRAL_FORCEINLINE __attribute__((always_inline)) inline
    #elif defined(_MSC_VER)
      #define SPECTRAL_FORCEINLINE __forceinline
    #else
      #define SPECTRAL_FORCEINLINE inline
    #endif
  #else
    #define SPECTRAL_FORCEINLINE inline
  #endif
#endif

/*
 * Maybe-unused annotation for local helper functions/variables.
 */
#ifndef SPECTRAL_MAYBE_UNUSED
  #if defined(__GNUC__) || defined(__clang__)
    #define SPECTRAL_MAYBE_UNUSED __attribute__((unused))
  #else
    #define SPECTRAL_MAYBE_UNUSED
  #endif
#endif

/*
 * Thread-local storage annotation
 */
#ifndef SPECTRAL_THREAD_LOCAL
  #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define SPECTRAL_THREAD_LOCAL _Thread_local
  #elif defined(__GNUC__) || defined(__clang__)
    #define SPECTRAL_THREAD_LOCAL __thread
  #else
    #define SPECTRAL_THREAD_LOCAL
  #endif
#endif

#endif /* SPECTRAL_MACROS_H */
