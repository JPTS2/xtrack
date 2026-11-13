<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# LEP: orbit, spin and polarization

Follow detector fields through coupled orbital optics and invariant spin fields,
then inspect what installed spin bumps and skew corrections recover.

| Example | Lesson |
| --- | --- |
| [001_orbital_and_spin_operating_point.py](001_orbital_and_spin_operating_point.py) | Compare the planar ring and detector-solenoid setting: orbital coupling, spin tune and invariant-axis tilt |
| [002_orbit_and_coupling_response.py](002_orbit_and_coupling_response.py) | Vary one installed spin bump with and without its paired skew correction; distinguish orbit, coupling and spin responses |
| [003_polarization_recovery.py](003_polarization_recovery.py) | Apply the full installed bump/correction settings and compare predicted equilibrium polarization and rate components |

The native model in
[`test_data/spin_refs_bmad`](../../../test_data/spin_refs_bmad/) represents a
45.585 GeV/c positron ring. Its bend fields and edges were simplified for Bmad
comparisons; it is an illustrative LEP model, not an operational machine setup.
The stored reference supplies mass, positive charge and magnetic-moment anomaly.
No external tracking code is required.

The calculations use fixed-energy 4D optics. The final lesson uses radiation
integrals to predict equilibrium polarization and polarization/depolarization
time constants; it does not track polarization build-up. The invariant axis has
an arbitrary sign. Acute tilt and the physical polarization projection along
the local bending field do not depend on that sign. The planar
Sokolov–Ternov direction is parallel to the magnetic field for positrons.

The correction settings are already installed in the model. Scaling them is not
rematching, and minimizing orbital coupling alone does not establish a
polarization optimum. The stored comparison model simplifies the bend fields
and edges; its analytical polarization predictions are not operational LEP
performance estimates.

For the underlying operations, see [spin](../../spin/README.md).
