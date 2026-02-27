# What Is It?

**Spectral_Engine** performs **spectral synthesis** on audio signals, either **offline** or in **realtime**. My goal is a **"synthesis from first principles"** engine, where **Additive**, **Subtractive**, and **Wavelett** (and most other synthesis forms) can be derived directly in the **spectral / STFT domain**. The core bet is simple: if we keep sound design in one coherent DSP domain, the whole system becomes easier to reason about, optimize, and extend.

# Ideal System Requirements / Performance Profile

Many of the sub-components in our signal chain are embarassingly parallel (which is especially true of offline mode.) We make use of SIMD intrinsics & GPU compute especially which are important to have for decent performance.

Much of the architecture is memory-bandwidth limited & prefers being resident in DRAM or cache rather than being streamed in, so having a good amount (16GB+) of fast memory (DDR5!) with a relatively modern CPU is required to saturate compute resources. This probably means that you will find hardware from 2020 onwards to be suitable.

| Component | Recommended Target | Minimum Target | Why It Matters |
|---|---|---|---|
| CPU SIMD | **AVX2 + FMA** | **SSE4.2** | Accelerates FFT, tracking, and inner-loop math |
| CPU Cores | **8 physical cores** | **4 physical cores** | Improves analysis parallelism and build throughput |
| CPU Frequency | **4.5 GHz+ boost** | **3.4 GHz+ boost** | Helps latency-sensitive analysis and scheduling paths |
| GPU | **Discrete GPU, 8 GB VRAM+** (CUDA-capable on Linux for CUDA path) | **None required** (CPU/export backend is enough) | GPU mainly helps dense synthesis and extreme voice counts |
| System Memory | **32 GB RAM** | **8 GB RAM** | Prevents memory pressure with large segment/control sets |
| Memory Bandwidth | **Dual-channel DDR5-5200+** or strong DDR4 | **Dual-channel DDR4-2666+** | Reduces stalls in bandwidth-heavy stages |
| Storage | **NVMe SSD** | **SATA SSD** | Speeds up build/cache/output workflow |

> [!INFO]
> **Minimum** means the smallest practical host to **render a segment binary offline and upload it to an embedded target**.  
> It is **not** intended for aggressive desktop runs with extreme segment density.  
> **Recommended** is a respectable **middle-ground desktop target** for high-throughput experimentation.
> This repository explicitly tries to keep the minimum spec in lockstep with an **RP4**, which would be a good choice of device to pair with an embedded target for offline segment processing.
> The repository reccomends using an **RP5** to get closer to the recommended spec.

<details>
<summary><strong>Info: Example Intensive Desktop Run</strong></summary>

> [!INFO]
> On one high-end desktop a very intensive configuration used  
> $N=4096$, $H=8$, and $\frac{N}{H}=512$ on a $\sim 40\,\mathrm{s}$ WAV containing complex speech.
>
> (This is with 0 overlapping frames, use of the most intensive window, most significantly, it is at a totally naive noise floor with amp detection set to -85dB, I.e., segments=117958342!)
>
> ```shell
> ./build/bin/spectral_x86_64_cuda_desktop ./resources/testing/shakespeare_he_saw_the_cat.wav 0 1 0 4096 8 -85 20 0
> ```
>
> ```shell
> FFT: 119.1ms  Track: 932.5ms  Synth: 155.2ms  Norm: 0.5ms  Write: 3.0ms  Total: 1210.3ms
> Audio: 39.13s  Realtime: 32.3x  Segs/sec: 97466K
> Memory: RSS 12718 MB, Peak tracked 7212.8 MB
> ```
>
> A still very heavy but far more reasonable load (increased hop-size to 128 from 8):
> ```shell
> ./build/bin/spectral_x86_64_cuda_desktop ./resources/testing/shakespeare_he_saw_the_cat.wav 0 1 0 4096 128 -85 20 0
> ```
>
> ```shell
> FFT: 19.3ms Track: 43.6ms Synth: 5.4ms Norm: 0.5ms Write: 7.4ms Total: 76.1ms
> Audio: 39.13s Realtime: 513.8x Segs/sec: 55837K
> Memory:  RSS 572 MB, Peak tracked 272.7 MB
>```
</details>

<details>
<summary><strong>Info: Full Benchmark Report (Example)</strong></summary>

> [!INFO]
> 4070ti, i5-13600k, 32GB DDR5 @ 5600MHz
> Example of full run:
> ```shell
> make clean && make clean-output && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPECTRAL_PRODUCTION_BUILD=OFF -DSPECTRAL_REPRO_BUILD=OFF && cmake --build build/ --target desktop --parallel
> ./build/bin/spectral_x86_64_cuda_desktop ./resources/testing/shakespeare_he_saw_the_cat.wav 0 1 0 4096 128 -85 20 0 # Short-run result
> PYTHONPATH=tools python3 -m spectral_tools.testing.benchmark_workflow bench -b build/bin/spectral_x86_64_cuda_desktop -i resources/testing/shakespeare_he_saw_the_cat.wav -P -- 0 1 0 4096 128 # Benchmark result
> ```

*The following is a dump from the benchmark tool for the hop size 128 case:*
```text
Context
repo_root: /home/tophurt/Desktop/Spectral_Engine
python: /home/tophurt/Desktop/Spectral_Engine/.venv/bin/python3
platform: Linux-6.12.73+deb13-amd64-x86_64-with-glibc2.41
machine: x86_64
perf_bin: /usr/bin/perf
perf_event_paranoid: 3
uname: Linux 6.12.73+deb13-amd64 x86_64 GNU/Linux
nproc: 20
git_head: 7978399f7e

Summary
tests: total=1 ok=1 failed=0 skipped=0
-P: ok=1
total_median_ms: n=1 min=73.200 median=73.200 mean=73.200 max=73.200
track_median_ms: n=1 min=43.900 median=43.900 mean=43.900 max=43.900
synth_kernel_median_ms: n=1 min=5.000 median=5.000 mean=5.000 max=5.000
synth_wall_median_ms: n=1 min=104.427 median=104.427 mean=104.427 max=104.427

Results
  single                             ok  -P=ok  total_median=73.200ms, track_median=43.900ms, total_warm_median=75.500ms, synth_kernel=5.000ms, synth_wall=104.427ms, track_bw=22.530GiB/s
    |- binary: /home/tophurt/Desktop/Spectral_Engine/build/bin/spectral_x86_64_cuda_desktop
    |- input: /home/tophurt/Desktop/Spectral_Engine/resources/testing/shakespeare_he_saw_the_cat.wav
    |- runs: 6
    |- mode: normal
    |- bench_args
    |  |- [0]: 0
    |  |- [1]: 1
    |  |- [2]: 0
    |  |- [3]: 4096
    |  `- [4]: 128
    |- metrics
    |  |- summary
    |  |  |- first_ms: 65.700
    |  |  |- median_ms: 73.200
    |  |  |- mean_ms: 74.616
    |  |  |- warm_median_ms: 75.500
    |  |  `- warm_mean_ms: null
    |  |- stage_medians
    |  |  |- fft_ms: 18.600
    |  |  |- track_ms: 43.900
    |  |  |- synth_kernel_ms: 5.000
    |  |  |- synth_wall_ms: 104.427
    |  |  |- synth_ms: 5.000
    |  |  `- norm_ms: null
    |  |- bandwidth_medians
    |  |  |- fft_gibps: 55.165
    |  |  |- fft_rel_pct: 100.000
    |  |  |- track_gibps: 22.530
    |  |  `- track_rel_pct: 40.844
    |  |- memory
    |  |  |- rss_median_mb: 630.339
    |  |  |- rss_mean_mb: 630.358
    |  |  |- rss_warm_median_mb: 630.351
    |  |  `- rss_warm_mean_mb: 630.364
    |  `- track_series
    |     |- count: 6
    |     |- median_ms: 43.900
    |     |- warm_median_ms: 44.100
    |     |- mean_ms: 45.350
    |     |- stdev_ms: 3.279
    |     |- min_ms: 42.200
    |     |- max_ms: 50.600
    |     |- outliers_iqr
    |     |  |- q1: 42.725
    |     |  |- q3: 47.850
    |     |  |- iqr: 5.125
    |     |  |- low_fence: 35.037
    |     |  |- high_fence: 55.537
    |     |  `- count: 0
    |     `- bw_median_gibps: 22.530
```
</details>


# Description / Topology

The top-level topology of the system is:

`INPUT -> STFT / Wavelett transform (or other analysis function) -> PARTIAL / SEGMENT CULL -> (RE)SYNTHESIS -> OUTPUT`

## STFT (Or DWT, WVD, etc.)

These are the key transforms we deal with. The **FFT**, by itself, is structurally insufficient to embed multiple layers of sound; it is fundamentally a **frequency transform**. With FFT we lose a dimension of data, so overlapping frequency/magnitude content gets mixed down into one complex value per bin.

By comparison, the **STFT** is a rank-2 **time-frequency tensor** formed by running FFT windows across time, giving us much stronger control for synthesis and editing.

*Why does this matter?*

Losing that dimension limits how we can manipulate sound, and FFT is entropic in this sense: it has virtually **zero temporal resolution**. If we imagine recording drums (strong transients and rapidly modulating components), FFT acts like an aggregate measure that can lose attack detail and bleed information into adjacent bins.

Other transforms make different temporal/frequency tradeoffs. **Wavelett** transforms are a configurable subset of **Linear Time Frequency Representations (LTFR)**. We also have **WVD (Wigner Ville Distribution)**, which has the theoretical minimum tradeoff between time and frequency resolution due to **Heisenberg's uncertainty principle**, but comes at significant computational cost. I still like WVD conceptually because it ties DSP back into physics in a very direct way.

This is not just theoretical; you can usually hear the difference, but more importantly you get a much clearer mental model for sound-design decisions.

*What about the rest of the chain?*

The particle of this system is called a **partial**. A partial is one of many waves that, when summed in its phase-band, produce the frequency response of the system. A group of partials lives in a **segment**. If we play back all partials in a segment by synthesizing those waves in parallel, we play back the post-transform sound.

To synthesize each partial uniquely, we need a voice per partial. In analogue terms, a voice is roughly **VCO + VCA + VCF (+ optional downmix)**.

## Partial & Segment Culling

*Conceptually simple, implementation-wise often not.* After transform, we usually have a large amount of harmonically unrelated or irrelevant content. The signal has a **noise floor** that gates artefacts (imperceptible/noisy partials) from meaningful features, especially in speech and distinct timbres.

Here, "irrelevant" is an operational term. Many stochastic, statistical, and dynamic models can bound signal loss tightly, which saves work later during synthesis. On the other hand, like JPEG, some heuristics intentionally trade audio quality for performance. In practice, this is pruning an underdetermined contribution relative to psychoacoustic perception.

Spectral synthesis can be demanding because we are often using thousands or millions of voices simultaneously, and each voice is effectively **VCO + VCA + downmix + VCF**. Smart culling reduces this workload significantly.

## (Re)Synthesis

This is where it gets fun. We can treat each partial as a data point because, up to the particle level, we can (re)synthesize it however we want: arbitrary waveform, envelope, LFO control, etc. By default the engine batches partials into master segments for layered control.

In the same way some synths have routing matrices, this engine has a much larger matrix with fine control down to each partial. The denser this matrix, the more the system becomes **memory-bandwidth limited**. This is why grouping partials into master segments is preferred for a sparser control matrix.

The engine has advanced control options for experimental design. Master partials can be grouped arbitrarily (including non-contiguous phase bins), as contours, or by harmonic relations in the spectrogram. With algorithms that account for interactions between particles, we can shape content harmonically and musically in very flexible ways.

# Features

- [ ] Multiple methods for spectral analysis
  - Wavelett bases for accelerated/custom applications
  - Other bases with special implications for sound or synthesis design
- [ ] Resynthesis using standard or arbitrary oscillators
  - Complex oscillator design per segment group
  - High granularity for segment groups / frequency bins
  - Envelope design per voice
- [ ] Essentially unlimited polyphony count
- [ ] GPU acceleration + OpenMP support
- [ ] O.S. functionality or bare metal for embedded targets

## Warnings & Disclaimers

> [!CAUTION]
> This codebase is still in **PRE_ALPHA** release.
> It is entirely a WIP, many features are not yet implemented. Build & Run with caution.

<details>
<summary><strong>Warning: Transform choices have large compute tradeoffs</strong></summary>

> [!WARNING]
> Some high-resolution transforms (for example, WVD) can significantly increase runtime and memory cost compared with STFT-class workflows.
</details>

<details>
<summary><strong>Warning: Culling is lossy when pushed aggressively</strong></summary>

> [!WARNING]
> Heuristic or aggressive partial culling can alter transients, timbre, and fine detail. Tune thresholds for your target quality/performance envelope.
</details>

<details>
<summary><strong>Warning: Desktop-only QoL features on embedded targets may break behavior</strong></summary>

> [!WARNING]
> The full desktop build includes offline-processing and QoL features not always suitable for embedded constraints. Forcing them on embedded targets may be unstable.
</details>
