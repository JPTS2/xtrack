<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Longitudinal matching

Match RF controls to longitudinal optics and synchronous-orbit targets. Each
script loads a fresh model and runs independently.

| Example | Lesson |
| --- | --- |
| [001_match_synchrotron_tune.py](001_match_synchrotron_tune.py) | Match the positive synchrotron tune using cavity voltage |
| [002_match_phase_and_energy.py](002_match_phase_and_energy.py) | Match a static off-reference synchronous orbit using laboratory RF frequency and phase |

Both start from the [native RF ring](../../../test_data/example_lattices/README.md#rf-ring-qualification):
a 40 m ring with a 1 GeV/c electron reference, one thin cavity at harmonic 20,
100 kV and synchronous phase pi. The public cavity convention uses `abs(q0)`.
There is no radiation, acceleration, collective force or aperture model.

`001` varies voltage at fixed reference momentum, frequency and phase. It
matches `qs`, which is positive even though the signed longitudinal phase
advance decreases around this above-transition ring. Recomputed 6D optics
and synchronous-particle tracking check the solution.

`002` uses an absolute laboratory RF clock and searches for the synchronous
revolution period. Frequency selects the off-reference equilibrium momentum;
phase controls its longitudinal position on the selected stable branch.
The reference momentum stays fixed. This is a static equilibrium with zero
net energy gain per turn, rather than an acceleration programme.

The ordinary relative-time cavity map cannot supply that frequency-steering
lesson: changing frequency there changes the RF wavelength without changing
the turn clock. With the laboratory clock, the searched orbit satisfies
`frequency * t_rev = harmonic`. Its zeta slips relative to the reference clock;
that slip measures the difference between synchronous and reference periods.

See [longitudinal optics](../../optics/longitudinal/) for coordinate, emittance
and revolution-period conventions, and [RF examples](../../README.md#rf) for
the dynamics, timing and acceleration plan.
