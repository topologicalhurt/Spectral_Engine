#ifndef STUB_MATH_H
#define STUB_MATH_H
double sin(double); double cos(double); double sqrt(double); double floor(double); double fabs(double);
float sinf(float); float cosf(float); float sqrtf(float); float floorf(float); float fabsf(float); float fmodf(float,float); float atan2f(float,float); float log2f(float); float exp2f(float); float powf(float,float);
#define isfinite(x) __builtin_isfinite(x)
#define isnan(x) __builtin_isnan(x)
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
#endif
