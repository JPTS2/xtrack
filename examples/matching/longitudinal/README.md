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
| [002_match_phase_and_energy.py](002_match_phase_and_energy.py) | Match a prescribed synchronous momentum and arrival coordinate |
| [003_infer_rf_frequency_from_orbit.py](003_infer_rf_frequency_from_orbit.py) | Infer one frequency trim from a synthetic dispersive BPM measurement |

All three load the [demonstration ring](../../../test_data/example_lattices/demo_ring/):
a 1500 m ring with a 1 GeV/c electron reference and a thin cavity `rf.1` at
s=250 m, harmonic 200. Each enables 100 kV at the stable phase pi before
matching. The public cavity convention uses `abs(q0)`.
There is no radiation, acceleration, collective force or aperture model.

`001` varies voltage at fixed reference momentum, frequency and phase. It
matches `qs=0.010`, which is positive even though the signed longitudinal phase
advance decreases around this above-transition ring. Recomputed 6D optics
and synchronous-particle tracking check the solution.

`002` uses an absolute laboratory RF clock and searches for the synchronous
revolution period. Frequency selects the off-reference equilibrium momentum;
phase controls its longitudinal position on the selected stable branch.
The target is delta=1e-4 and entrance zeta=3 mm. The lab phase offset uses the
actual cavity position, not the circumference. Reference momentum stays fixed.
This is a static equilibrium with zero net energy gain per turn, rather than
an acceleration programme.

`003` uses a synthetic horizontal position at `bpm.s1.d` to infer
the known frequency trim, holding RF phase fixed. This BPM has nonzero horizontal
dispersion, which makes the small trim identifiable; a zero-dispersion position
can leave its sign ambiguous. The recovered trim, orbit residual and synchronous
momentum are independently recomputed. No measurement noise or unknown
alignment/calibration error is modelled.

The ordinary relative-time cavity map cannot supply that frequency-steering
lesson: changing frequency there changes the RF wavelength without changing
the turn clock. With the laboratory clock, the searched orbit satisfies
`frequency * t_rev = harmonic`. Its zeta slips relative to the reference clock;
that slip measures the difference between synchronous and reference periods.

See [longitudinal optics](../../optics/longitudinal/) for coordinate, emittance
and revolution-period conventions, and [RF examples](../../rf/README.md) for
tracking dynamics and the timing and acceleration routes.
