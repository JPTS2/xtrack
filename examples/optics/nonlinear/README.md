<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Nonlinear optics

Chromatic optics starts this route: first-order momentum dependence, followed
by higher-order chromaticity. Amplitude-dependent and other nonlinear studies
follow in the [collection structure](../../README.md#nonlinear).

Chromaticity describes momentum dependence. At each fixed momentum, the Twiss
calculation is still linearized about its closed orbit. Keeping first-order and
higher-order chromaticity together gives the subject one learning sequence.

| Example | Lesson |
| --- | --- |
| [001_chromatic_optics.py](001_chromatic_optics.py) | Inspect tune and local-optics derivatives and compare a momentum scan with linear chromaticity |
| [002_higher_order_chromaticity.py](002_higher_order_chromaticity.py) | Fit a momentum scan, interpret higher derivatives and compare fit-window and polynomial-order choices |

These examples use the [native magnetic ring](../../../test_data/example_lattices/README.md)
with a 1 GeV/c electron reference. Reference momentum and magnet strengths stay
fixed while `delta` varies. There is no RF, radiation or collective force.

Linear chromaticities are `dQ/d(delta)`. In each uncoupled plane, the local
functions are `b = (d beta/d delta)/beta`,
`a = d alpha/d delta - alpha*b`, and `W = sqrt(a^2+b^2)`.
Finite-difference checks have truncation and numerical errors.

`002` uses `line.get_non_linear_chromaticity`, which performs 4D Twiss scans and
fits their tunes. `dnqx[n]` and `dnqy[n]` are the fitted derivatives at
`delta = 0`, including the factorial: `Q(delta) = sum(dnQ[n]*delta^n/n!)`.
Order zero is a fitted intercept. A fit over a finite range need not equal
the local Twiss derivatives exactly. The stored Twiss samples also let us
compare a quadratic orbit coefficient with second-order dispersion `ddx`.

The lesson repeats the fit with a narrower range and a higher degree, then
compares predictions over their shared window. Numerical precision, the scan
range and fit degree all matter. The qualified scan stays on a stable optical
branch; this helper does not unwrap tune branches or determine momentum
acceptance. Higher-order estimates need their own convergence check.

For closed-orbit momentum shifts and dispersion, see
[transverse optics](../transverse/README.md). Chromaticity correction belongs
with matching.
