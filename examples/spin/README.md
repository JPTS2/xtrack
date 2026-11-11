<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Spin

Follow the sequence from single-element precession to ring spin dynamics and
radiative polarization. Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_spin_precession.py](001_spin_precession.py) | Track a radial spin through a sector bend and compare with Thomas–BMT precession |
| [002_closed_spin_solution.py](002_closed_spin_solution.py) | Find the periodic spin direction and show how a solenoid changes it |
| [003_track_spin_in_a_ring.py](003_track_spin_in_a_ring.py) | Separate individual precession from dephasing across a finite momentum sample |
| [004_spin_tune_vs_energy.py](004_spin_tune_vs_energy.py) | Scan the native fractional spin tune and compare with the folded planar relation |
| [005_resonant_depolarization.py](005_resonant_depolarization.py) | Compare resonant and detuned magnetic drives, distinguishing spin rotation from ensemble dephasing |
| [006_equilibrium_polarization.py](006_equilibrium_polarization.py) | Inspect signed electron equilibrium polarization and its spin-orbit dependence |
| [007_radiative_depolarization.py](007_radiative_depolarization.py) | Compare mean transport with photon-recoil-driven loss of spin coherence |
| [008_invariant_spin_field.py](008_invariant_spin_field.py) | Launch spins along the native linear invariant field away from the closed orbit |

Spin components use the local horizontal, vertical and longitudinal axes.
Named electron references supply the magnetic-moment anomaly; the initial spin
vector is set explicitly. The invariant axis `n0` and `-n0` describe the same
axis. Physical equilibrium polarization is `spin_polarization_eq * n0` and does
not depend on that choice of sign.

`002`–`008` use the shared
[demonstration ring](../../test_data/example_lattices/demo_ring/README.md).
Most use a 10 GeV/c reference; `004` scans momentum and `007` chooses a nearby
spin-sensitive operating point. Fixed normalized magnet strengths imply
physical fields scaled with reference rigidity. These are illustrative settings.

`001`–`005` and `008` leave RF and radiation off. `006` and `007` use native
mean-energy-loss compensation and magnet tapering, introduced in
[synchrotron radiation](../radiation/README.md).
`006` evaluates the signed physical polarization along the magnetic field:
electrons polarize antiparallel to it in the planar Sokolov–Ternov limit.

A falling vertical spin projection can be coherent rotation.
A falling magnitude of the ensemble mean instead shows loss of coherence.
The finite tracking samples in `003`, `005` and `007` do not determine a
polarization lifetime. In `007`, photon recoil changes orbital motion and
therefore BMT precession; individual spin vectors retain their length.
Native polarization analysis in `006` predicts equilibrium polarization and
polarization/depolarization time constants. These processes are not simulated
by the tracking comparison.

`008` extends the periodic-axis lesson to the orbital dependence of the spin
field, using the native `spin_n_matrix` and two small-amplitude probes.
