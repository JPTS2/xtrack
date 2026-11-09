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
| [003_amplitude_detuning.py](003_amplitude_detuning.py) | Measure four action-detuning coefficients and compare tracked phases with the action-frequency relation |
| [004_multiturn_optics.py](004_multiturn_optics.py) | Find a nonzero four-turn island orbit, compute its local optics and compare it with tracking |
| [005_stability_diagram.py](005_stability_diagram.py) | Build a tracked action footprint and evaluate the Gaussian dispersion integral |

`001` and `002` use the [native magnetic ring](../../../test_data/example_lattices/README.md)
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

`003` and `005` use explicit integrable detuning maps, with symmetric cross
coefficients compatible with a common action Hamiltonian. Geometric actions
are in metres, with `J = amplitude_sigma^2 * nemitt/(2*beta0*gamma0)`; the
factor of two is part of the convention. The emittances set action scales.
These maps isolate the APIs and their assumptions from a machine lattice.

`003` requires optional **nafflib** for
`line.get_amplitude_detuning_coefficients`. Its independent phase measurement
checks the resulting action-frequency relation. Tunes remain below one half
in the qualified scan; other tune branches and finite records require suitable
frequency estimation. `005` uses FFT extraction and needs no nafflib.

`004` combines a thin octupole with a linear rotation. Its analytic seed is
specifically a period-four orbit; the four-turn search and its local phase
advance describe that island, rather than the zero core orbit. The chosen
parameters give stable local optics. A different resonance or turn count needs
its own seed and stability qualification.

`005` uses the tracked uniform-action grid with an uncoupled Gaussian action
density. The returned boundary follows the API's positive-imaginary sign
convention. The finite regulator, action cutoff and quadrature are explicit;
the example does not establish convergence or supply an impedance/coherent-mode
model for a stability assessment.

For closed-orbit momentum shifts and dispersion, see
[transverse optics](../transverse/README.md). Chromaticity correction belongs
with matching.
