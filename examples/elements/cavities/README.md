<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Cavities

Learn the single-element RF kicks and transport maps available in Xtrack.
Each script runs independently. Phases are in radians, frequencies in hertz
and voltages in volts. The examples use protons with reference charge `q0 = +1`.

| Example | Lesson |
| --- | --- |
| [001_cavity_kick.py](001_cavity_kick.py) | Energy kick versus phase, arrival coordinate and voltage; fixed reference momentum |
| [002_finite_length_cavity_transport.py](002_finite_length_cavity_transport.py) | Energy, time of flight, relative RF phase and transverse slope through a finite-length cavity |
| [003_cavity_models.py](003_cavity_models.py) | Query and compare the public numerical map choices with a common kick count |
| [004_transverse_rf_kicks.py](004_transverse_rf_kicks.py) | CrabCavity/RFMultipole conventions and the companion off-axis energy kick |

`001` uses a zero-length cavity and a 1 GeV/c reference. With the relative RF
clock, the proton energy kick is `V*sin(phase - 2*pi*f*zeta/(beta0*c))` in eV.
Positive `zeta` means earlier arrival. Tracking updates particle energy, `delta`
and `ptau` while leaving reference `p0c` unchanged.

`002` uses a 200 MeV/c reference to make velocity changes visible. It compares
voltage off/on at the same length, using the exact drift treatment and uniform
RF kick/drift steps. Standalone element tracking retains the exit `s`, allowing
elapsed flight time to be calculated from `(Delta s-Delta zeta)/(beta0*c)`. The phase column is
the relative-clock phase at each endpoint. Acceleration changes longitudinal
momentum and geometrical slope while preserving transverse canonical momentum.

The finite-length public `Cavity` map distributes longitudinal kicks between
transport steps. It describes this kick/drift model, without introducing an
RF field map, cavity-wall geometry or additional RF focusing lenses. `003`
compares the available transport approximations at fixed length, voltage,
integrator and kick count. “Exact” in the baseline name refers to drift
transport; the comparison is not an RF-field accuracy or convergence ranking.

`004` uses zero-length elements with a 1 GeV/c reference. `CrabCavity` uses a
sine phase convention; the normal `RFMultipole` dipole uses a negative cosine
kick. Matching them requires `knl[0] = crab_voltage/p0c` and a `pi/2` offset in
`phase_n[0]` for this unit-charge reference. An off-axis launch exposes the
associated longitudinal kick as well as the transverse deflection.

These examples use the relative RF clock. For longitudinal dynamics and the
laboratory-time RF and acceleration routes, see
[RF and longitudinal motion](../../rf/README.md). For model assembly, see
[line construction](../../line_construction/README.md).
