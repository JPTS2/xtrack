<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Nonlinear tracking

Measure amplitude-dependent frequencies, then sample finite-turn survival.
Each script loads a fresh model and runs independently.

| Example | Lesson |
| --- | --- |
| [001_tune_footprint.py](001_tune_footprint.py) | Track normalized-amplitude launches and extract a transverse tune footprint |
| [002_dynamic_aperture.py](002_dynamic_aperture.py) | Map surviving and lost launches over a stated tracking horizon |

Both use the installed octupole circuits in the
[shared demonstration ring](../../../test_data/example_lattices/demo_ring/README.md):
1.5 km circumference and a 1 GeV/c electron reference. The examples power
`k3lf` at `oct.f`; `k3ld` at `oct.d` stays zero by default. Geometry is unchanged,
RF voltage is zero, and radiation and collective forces are absent. Normalized
emittances are in metres, with geometric emittance `nemitt/(beta0*gamma0)`.
The launch amplitudes and phases are part of each study. Both use integrated
strength `k3lf=3e4 m^-3` and `k3ld=0`. The footprint uses a conservative surviving
amplitude grid; the survival study extends to larger amplitudes and includes
a field-off control.

`001` uses the public FFT footprint calculation. Its real-signal spectra return
dominant frequencies folded into `[0, 0.5]` cycles per turn. The small-amplitude
nominal tunes near `Qx=8.7548`, `Qy=10.0175` therefore appear near
`0.2452`, `0.0175`.
These results do not recover integer tunes or automatically identify modes and
resonance harmonics. Increasing the zero-padded FFT length refines the frequency
grid; it does not add tracked information or establish tune convergence.

The RF-free footprint uses numerical longitudinal freezing to prepare 4D
launches and keep momentum fixed. The CPU context permits local compilation
of the frozen kernel, requiring a working C compiler. Every footprint launch
must survive; lost trajectories are not assigned valid tunes by this helper.

`002` reports the survival of a specified launch grid, momentum offset, phases
and number of turns. Particle identifiers associate results with the original
launches after loss-driven reordering. A short surviving trajectory does not
establish infinite-time stability, and a sampled boundary need not be monotonic
or independent of launch phase. Numerical escape guards are not chamber walls.

For detuning observables and nonlinear optics analysis, see
[nonlinear optics](../../optics/nonlinear/). Physical aperture geometry and loss
refinement belong to [apertures](../../README.md#apertures).
