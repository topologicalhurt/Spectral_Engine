# Academic Source Basis

Only academic/technical sources are used for this research plan. Vendor manuals may be useful during implementation, but they are not the basis of the mathematical claims below.

## FFT, STFT, and spectral analysis

1. J. W. Cooley and J. W. Tukey, “An Algorithm for the Machine Calculation of Complex Fourier Series,” *Mathematics of Computation*, 1965.
2. A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
3. J. G. Proakis and D. G. Manolakis, *Digital Signal Processing: Principles, Algorithms, and Applications*, 4th ed., Pearson, 2007.
4. F. J. Harris, “On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform,” *Proceedings of the IEEE*, 1978. DOI: https://doi.org/10.1109/PROC.1978.10837
5. D. W. Griffin and J. S. Lim, “Signal Estimation from Modified Short-Time Fourier Transform,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 1984.
6. M. Dolson, “The Phase Vocoder: A Tutorial,” *Computer Music Journal*, 1986.
7. J. O. Smith, *Spectral Audio Signal Processing*, W3K Publishing, 2011. QIFFT notes: https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html

## Sinusoidal modeling and additive resynthesis

8. R. J. McAulay and T. F. Quatieri, “Speech Analysis/Synthesis Based on a Sinusoidal Representation,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 1986.
9. X. Serra and J. O. Smith, “Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based on a Deterministic plus Stochastic Decomposition,” *Computer Music Journal*, 1990.
10. M. Goodwin, “Adaptive Signal Models: Theory, Algorithms, and Audio Applications,” Springer, 1998.
11. J. Laroche and M. Dolson, “Improved Phase Vocoder Time-Scale Modification of Audio,” *IEEE Transactions on Speech and Audio Processing*, 1999.

## Frequency estimation and peak interpolation

12. D. C. Rife and R. R. Boorstyn, “Single Tone Parameter Estimation from Discrete-Time Observations,” *IEEE Transactions on Information Theory*, 1974. DOI: https://doi.org/10.1109/TIT.1974.1055282
13. B. G. Quinn, “Estimating Frequency by Interpolation Using Fourier Coefficients,” *IEEE Transactions on Signal Processing*, 1994. DOI: https://doi.org/10.1109/78.295186
14. E. Jacobsen and P. Kootsookos, “Fast, Accurate Frequency Estimators,” *IEEE Signal Processing Magazine*, 2007. DOI: https://doi.org/10.1109/MSP.2007.361611
15. C. Candan, “A Method for Fine Resolution Frequency Estimation from Three DFT Samples,” *IEEE Signal Processing Letters*, 2011. DOI: https://doi.org/10.1109/LSP.2011.2136378

## Numerical accuracy and floating point

16. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
17. D. Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic,” *ACM Computing Surveys*, 1991.
18. W. Kahan, “Further Remarks on Reducing Truncation Errors,” *Communications of the ACM*, 1965.
19. IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019. Source: https://standards.ieee.org/standard/754-2019.html
20. NIST Digital Library of Mathematical Functions, inverse hyperbolic tangent logarithmic identity and series. Sources: https://dlmf.nist.gov/4.37.E25 and https://dlmf.nist.gov/4.37.E31

## Performance, cache, parallel systems, and real-time constraints

21. J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., Morgan Kaufmann, 2017.
22. D. A. Patterson and J. L. Hennessy, *Computer Organization and Design: The Hardware/Software Interface*, Morgan Kaufmann.
23. G. C. Buttazzo, *Hard Real-Time Computing Systems*, Springer, 2011.
24. M. Frigo and S. G. Johnson, “The Design and Implementation of FFTW3,” *Proceedings of the IEEE*, 2005.

## Fixed-point and embedded DSP

25. K. K. Parhi, *VLSI Digital Signal Processing Systems: Design and Implementation*, Wiley, 1999.
26. S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, McGraw-Hill.
27. R. Lyons, *Understanding Digital Signal Processing*, 3rd ed., Pearson, 2010.

## Inverse-FFT and parallel/GPU additive synthesis

28. X. Rodet and P. Depalle, “Spectral Envelopes and Inverse FFT Synthesis,” *Proc. AES 93rd Convention*, 1992. (The FFT⁻¹ additive-synthesis method — the basis of the F-stream IFFT path; ~15× over oscillator banks on CPU.)
29. A. Freed, X. Rodet, and P. Depalle, “Synthesis and Control of Hundreds of Sinusoidal Partials on a Workstation without Custom Hardware,” *Proc. ICSPAT*, 1992. (First real-time transform-domain synthesizer.)
30. L. Savioja, V. Välimäki, and J. O. Smith, “Real-Time Additive Synthesis with One Million Sinusoids Using a GPU,” *Proc. AES 128th Convention*, 2010; and “Audio Signal Processing Using Graphics Processing Units,” *J. Audio Eng. Soc.*, 2011. (Additive synthesis is embarrassingly data-parallel — the basis for the F6/F7 SIMDe/vDSP/GPU parallelization of the IFFT path.)

## How these sources constrain implementation

- Window functions and interpolation must distinguish coherent gain, RMS gain, and leakage behavior; see Harris and Oppenheim/Schafer.
- Sinusoidal resynthesis must preserve phase and instantaneous frequency units; see McAulay/Quatieri and Serra/Smith.
- Peak estimators must be chosen by estimator assumptions, not empirical constants; see Rife/Boorstyn, Quinn, Jacobsen/Kootsookos and Candan.
- Fast approximations need quantified numerical error and propagation analysis; see Higham, Goldberg, IEEE 754-2019 and the NIST DLMF identities used by the shared peak-log approximation.
- Hot-loop performance claims must be validated on target architectures; see Hennessy/Patterson and Frigo/Johnson.
- Embedded behavior must be bounded in memory and time; see Buttazzo and Parhi.
