<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Synchrotron radiation

Start with deterministic energy loss, compensate it, then study damping,
linear equilibrium predictions and stochastic photon emission. Each script
runs independently.

| Example | Lesson |
| --- | --- |
| [001_mean_energy_loss.py](001_mean_energy_loss.py) | Compare radiation off and mean energy loss during one uncompensated passage |
| [002_compensate_radiation_and_taper.py](002_compensate_radiation_and_taper.py) | Use native RF compensation and magnet tapering to find a radiating periodic operating point |
| [003_radiation_integrals.py](003_radiation_integrals.py) | Inspect native integral predictions and locate energy-loss and excitation contributions |
| [004_mean_radiation_damping.py](004_mean_radiation_damping.py) | Follow two amplitude damping times and compare action decay with the native rate |
| [005_equilibrium_emittances.py](005_equilibrium_emittances.py) | Inspect equilibrium emittances and the full native beam covariance |
| [006_quantum_radiation.py](006_quantum_radiation.py) | Contrast deterministic loss with photon-resolved and total-kick fluctuations |
| [007_record_photons_and_spectrum.py](007_record_photons_and_spectrum.py) | Record photons, inspect their spectrum and compare their energies with particle losses |

All seven examples use a 10 GeV/c electron reference. `001`–`005` load the shared
[demonstration ring](../../test_data/example_lattices/demo_ring/README.md).
Its input reference momentum is 1 GeV/c; selecting 10 GeV/c with fixed normalized
magnet strengths corresponds to ten times the nominal physical fields.
The powered ring examples use 50 MV at harmonic 200, with about 23.17 MeV lost
per turn. This is an illustrative operating point, without a hardware claim.

`001` leaves RF off. `002` introduces native RF compensation and magnet tapering;
`004` and `005` reuse that operation on fresh copies of the same ring.
Compensation restores the energy each turn, while tapering keeps the operating
optics close to the intended setting along the energy sawtooth. `003` uses
radiation-free optics for its integral predictions.

`004` follows about 2.08 amplitude damping times in 2,048 turns. At fixed
geometry, the amplitude damping rate scales approximately with the cube of
beam energy. The action decrement has twice the amplitude exponent, leaving
about 1.56% of the initial mode action. Mean tracking supplies damping without
random excitation;
it cannot produce the nonzero quantum equilibrium predicted in `005`.
That prediction comes from native damping/excitation analysis of the mean map.
This planar uncoupled model has zero vertical equilibrium emittance: collinear
photon recoil omits photon-opening-angle excitation. Numerical remainders do not
represent a physical beam-height floor.

Equilibrium calculations are linear predictions. They do not establish
nonlinear acceptance, survival or tails of a stored bunch.

`006` and `007` use explicit dipole probes at the same 10 GeV/c. `006` tracks a
finite-length `Bend`; `007` records a thin radiating `Multipole` kick with an
assigned bending length. The latter does not describe finite-length transport.
`quantum` generates individual photons, whereas `quantum-kick` samples a total
stochastic kick without photon records. Native tracking initializes the random
streams; stochastic realizations vary between runs.

For high-statistics distribution and tail comparisons, see
[quantum-kick versus quantum](quantum_kick_vs_quantum/README.md). Those studies
serve a different purpose from the short introductory tracking examples.
