<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Longitudinal dynamics

Follow oscillating particles, inspect the RF bucket, then explore freely
circulating probes and changing RF conditions. Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_synchrotron_motion.py](001_synchrotron_motion.py) | Track small longitudinal oscillations and compare with the linear synchrotron tune |
| [002_rf_bucket_and_separatrix.py](002_rf_bucket_and_separatrix.py) | Use the native bucket and separatrix to distinguish bounded motion from phase slip |
| [003_coasting_arrival_times.py](003_coasting_arrival_times.py) | Distinguish the shared clock from momentum-dependent particle arrival times |
| [004_transition_crossing.py](004_transition_crossing.py) | Change the accelerating RF phase branch as the slip factor changes sign |
| [005_slip_stacking.py](005_slip_stacking.py) | Follow two populations associated with different absolute-time RF frequencies |

`001` and `003` use the shared 1 GeV/c electron
[demonstration ring](../../../test_data/example_lattices/demo_ring/README.md).
`001` powers its RF cavity; `003` leaves RF voltage zero and launches transverse
closed orbits at small fixed momentum offsets. Those samples do not establish
the bare ring's momentum acceptance. Positive zeta means arrival before the
reference at a fixed observation plane.

`002` uses an explicit nonlinear `LineSegmentMap` and the public Xpart helper
`xp.longitudinal.get_bucket`. Its stationary conservative bucket describes that
ideal model, rather than a full lattice's transverse stability or radiation
equilibrium. The example requires **Xpart** for bucket construction.

`004` uses an accelerating proton turn map with a prescribed transition gamma.
The native reference-energy increment changes the reference momentum; the RF
settings select the appropriate phase branch. The comparison describes a finite
tracking experiment, without an adiabatic-capture or laboratory-clock claim.

`005` uses two absolute-time cavities and an RF-free map for momentum-dependent
slip, with a 26 GeV/c proton reference. Each RF system's synchronous period and
momentum are found separately, then both fields act on two small prescribed
populations. The isolated operating points are references, not fixed points of
the combined time-dependent system. The example shows relative slipping,
without a capture sequence or stacking-yield claim.

For RF element parameters, linear optics and matching, return to
[RF and longitudinal motion](../README.md).
