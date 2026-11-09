<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Longitudinal optics

Solve the synchronous orbit and three optical modes, specify longitudinal beam
covariance, then distinguish reference, synchronous and off-momentum periods.
Read in order; every script runs independently.

| Example | Lesson |
| --- | --- |
| [001_six_dimensional_twiss.py](001_six_dimensional_twiss.py) | Solve periodic 6D optics and verify the synchronous particle returns in all six coordinates |
| [002_longitudinal_covariance.py](002_longitudinal_covariance.py) | Set a longitudinal mode emittance, inspect correlations and draw a covariance contour |
| [003_revolution_period.py](003_revolution_period.py) | Compare reference and searched periods, then measure momentum-dependent periods without RF |

The [shared demonstration ring](../../../test_data/example_lattices/demo_ring/README.md)
uses a 1 GeV/c electron reference and a 1.5 km circumference. Each example
explicitly powers its thin cavity at 100 kV, harmonic 200 and phase pi.
This is stable in the current public `Cavity` convention using `abs(q0)`.
RF tracking is sinusoidal; Twiss is a local linearization around the synchronous
orbit. There is no radiation, acceleration, collective force or aperture model.

The canonical longitudinal pair is `(zeta, pzeta)`, with zeta in metres and
`pzeta = (E-E0)/(beta0*p0*c)` for the reference species. Positive zeta means an
earlier arrival at fixed position and reference clock time. `pzeta` and `delta`
agree to first order near the reference particle.
Longitudinal phase `muzeta` is signed and decreases around this above-transition
ring; the synchrotron tune `qs` is reported as a positive value.

`002` specifies geometric longitudinal emittance in metres, rather than eV*s.
Its target bunch length refers to the longitudinal mode's contribution at the
entrance. The full covariance includes all three mode populations and can have
additional longitudinal variance from transverse modes. It describes centered
linear moments; the contour is `u^T Sigma^-1 u = 1`, not an RF separatrix or a
statement that the full distribution is matched at finite amplitude.

`t_rev0` is the reference length divided by `beta0*c`; `t_rev` is supplied by
`search_for_t_rev=True`. This search repeats `x, px, y, py, delta` over sampled
turns while allowing zeta slippage to determine the period. Reference and
searched periods agree closely for the nominal synchronous model here.
A fresh independent load with RF voltage zero supports a fixed-delta study:
one-turn zeta slippage measures the actual period. Its first derivative is the
slip factor. Reference momentum and magnetic strengths stay fixed.

The off-momentum period sample uses `delta = -1e-4, 0, 1e-4`. The bare ring
has large chromaticity and a vertical tune near an integer; these points do not
qualify a momentum-acceptance interval.
