SPECTRAL - Additive Spectral Audio Processor

This is a real-time spectral analysis and resynthesis tool targeting the following platforms:

- LLVM software renderer (Cross-platform)
- Metal (MacOS)
- Cuda (Linux)

It extracts sinusoidal partials from audio via FFT peak tracking
and resynthesizes them with time stretching and pitch shifting.

It also currently supports additive resynthesis with some rudimentary
oscillator shapes.

BUILDING

  Requirements (macOS):
    - Xcode command line tools
    - Homebrew packages: libomp, libsndfile

  Requirements (Linux):
    - GCC with OpenMP support
    - FFTW3, libsndfile

  To build:
    cd spectral_engine
    make deps

USAGE

  ./spectral input.wav [timbre] [stretch] [pitch] [n_fft] [hop] [db_thresh] [threads] [backend]

  Defaults: timbre=0 stretch=1.0 pitch=0 n_fft=4096 hop=128 db_thresh=-85 threads=auto backend=auto

  Timbres: 0=sine 1=saw 2=square 3=triangle 4=asin 5=parabola 6=quantized 7=pwm

  Backends: 0=auto 1=cpu 2=metal 3=cuda 4=export

  Output is written to out_c.wav

PERFORMANCE

  Here is a program dump for metal:

```
Analyzing 1725440 frames (n_fft=8192, hop=1024, db_thresh=-100.0, threads=12)...
Found 1336294 segments.
Metal: Apple M1 Pro
Rendering with Metal GPU (tile-parallel, 1336294 segs, density=0.8)...
Metal: 1336294 segs, 6740 tiles, avg 991 segs/tile

--- Timing ---
FFT:       8.365 ms
Tracking:  22.318 ms
Synthesis: 58.186 ms
Normalize: 0.186 ms
Total:     89.055 ms

--- Throughput ---
Audio:     39.13 sec
Realtime:  439.3x
Segs/sec:  15005 K

--- Performance Metrics ---
Memory (physical):
  RSS:            286 MB
  Peak tracked:   94.7 MB
CPU Time:
  User:           119.8 ms
  System:         152.2 ms
  Total CPU:      272.0 ms
Utilization:
  Threads used:   12 / 8 cores
  Core util:      17.8% (of 12 threads)
  Parallelism:    2.13x effective

Done.
```

AUTHOR

  Connor Sinclair / topologicalhurt
  csin0659@uni.sydney.edu.au
