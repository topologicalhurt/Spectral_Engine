# Academic Source Basis

Only academic/technical sources are used for this research plan. Vendor manuals may be useful during implementation, but they are not the basis of the mathematical claims below.

## FFT, STFT, and spectral analysis

1. J. W. Cooley and J. W. Tukey, “An Algorithm for the Machine Calculation of Complex Fourier Series,” *Mathematics of Computation*, 1965.
2. A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
3. J. G. Proakis and D. G. Manolakis, *Digital Signal Processing: Principles, Algorithms, and Applications*, 4th ed., Pearson, 2007.
4. F. J. Harris, “On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform,” *Proceedings of the IEEE*, 1978.
5. D. W. Griffin and J. S. Lim, “Signal Estimation from Modified Short-Time Fourier Transform,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 1984.
6. M. Dolson, “The Phase Vocoder: A Tutorial,” *Computer Music Journal*, 1986.
7. J. O. Smith, *Spectral Audio Signal Processing*, W3K Publishing, 2011.

## Sinusoidal modeling and additive resynthesis

8. R. J. McAulay and T. F. Quatieri, “Speech Analysis/Synthesis Based on a Sinusoidal Representation,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 1986.
9. X. Serra and J. O. Smith, “Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based on a Deterministic plus Stochastic Decomposition,” *Computer Music Journal*, 1990.
10. M. Goodwin, “Adaptive Signal Models: Theory, Algorithms, and Audio Applications,” Springer, 1998.
11. J. Laroche and M. Dolson, “Improved Phase Vocoder Time-Scale Modification of Audio,” *IEEE Transactions on Speech and Audio Processing*, 1999.

## Frequency estimation and peak interpolation

12. D. C. Rife and R. R. Boorstyn, “Single Tone Parameter Estimation from Discrete-Time Observations,” *IEEE Transactions on Information Theory*, 1974.
13. B. G. Quinn, “Estimating Frequency by Interpolation Using Fourier Coefficients,” *IEEE Transactions on Signal Processing*, 1994.
14. E. Jacobsen and P. Kootsookos, “Fast, Accurate Frequency Estimators,” *IEEE Signal Processing Magazine*, 2007.
15. C. Candan, “A Method for Fine Resolution Frequency Estimation from Three DFT Samples,” *IEEE Signal Processing Letters*, 2011.

## Numerical accuracy and floating point

16. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
17. D. Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic,” *ACM Computing Surveys*, 1991.
18. W. Kahan, “Further Remarks on Reducing Truncation Errors,” *Communications of the ACM*, 1965.

## Performance, cache, parallel systems, and real-time constraints

19. J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., Morgan Kaufmann, 2017.
20. D. A. Patterson and J. L. Hennessy, *Computer Organization and Design: The Hardware/Software Interface*, Morgan Kaufmann.
21. G. C. Buttazzo, *Hard Real-Time Computing Systems*, Springer, 2011.
22. M. Frigo and S. G. Johnson, “The Design and Implementation of FFTW3,” *Proceedings of the IEEE*, 2005.

## Fixed-point and embedded DSP

23. K. K. Parhi, *VLSI Digital Signal Processing Systems: Design and Implementation*, Wiley, 1999.
24. S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, McGraw-Hill.
25. R. Lyons, *Understanding Digital Signal Processing*, 3rd ed., Pearson, 2010.

## How these sources constrain implementation

- Window functions and interpolation must distinguish coherent gain, RMS gain, and leakage behavior; see Harris and Oppenheim/Schafer.
- Sinusoidal resynthesis must preserve phase and instantaneous frequency units; see McAulay/Quatieri and Serra/Smith.
- Peak estimators must be chosen by estimator assumptions, not empirical constants; see Rife/Boorstyn, Quinn, Jacobsen/Kootsookos and Candan.
- Fast approximations need quantified numerical error and propagation analysis; see Higham and Goldberg.
- Hot-loop performance claims must be validated on target architectures; see Hennessy/Patterson and Frigo/Johnson.
- Embedded behavior must be bounded in memory and time; see Buttazzo and Parhi.
