<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Longitudinal dynamics

Follow synchrotron motion, inspect RF buckets under distinct operating
conditions, then study arrival delays, transition crossing and slipping trains.
Every script is independently runnable.

| Example | Lesson |
| --- | --- |
| [001_synchrotron_motion.py](001_synchrotron_motion.py) | Compare small longitudinal oscillations with the linear synchrotron tune |
| [002_rf_bucket_and_separatrix.py](002_rf_bucket_and_separatrix.py) | Compare librating and circulating trajectories with a stationary separatrix |
| [003_accelerating_bucket.py](003_accelerating_bucket.py) | Explain the tilted accelerating bucket and prepare a native matched sample |
| [004_radiating_bucket.py](004_radiating_bucket.py) | Inspect an effective bucket around a compensated radiating closed orbit |
| [005_coasting_arrival_times.py](005_coasting_arrival_times.py) | Distinguish reference clock and momentum-dependent particle arrival |
| [006_transition_crossing.py](006_transition_crossing.py) | Change the accelerating RF phase branch as the slip factor changes sign |
| [007_slip_stacking.py](007_slip_stacking.py) | Prepare two small-amplitude matched bunch trains and follow their slipping under both RF systems |

`001` and `005` use the shared 1 GeV/c electron
[demonstration ring](../../../test_data/example_lattices/demo_ring/README.md).
The first powers RF; the coasting example leaves it off and launches small
fixed momentum offsets. Those samples do not establish momentum acceptance.
Positive zeta means earlier arrival at a fixed observation plane.

## Buckets

`002` and `003` use an explicit nonlinear `LineSegmentMap` and the public
Xpart bucket entry point. The stationary model has a symmetric conservative
bucket; its outside-bucket particles slip in RF phase without being removed.
The accelerating model supplies 50 keV per turn to the reference, shifts the
RF phase and prepares a finite-length native matched sample. Its entrance
Hamiltonian is not assumed invariant under the accelerating discrete map.

`004` loads the shared ring at 10 GeV/c, powers 50 MV RF, compensates mean
radiation and applies native tapering. The effective bucket has an asymmetric
zeta range and shifted momentum centre. A short matched sample is shown on
both the full bucket scale and a local zoom. It is not a quantum-radiation
equilibrium distribution; damping prevents Hamiltonian conservation.
For compensation, see [radiation](../../radiation/README.md).

## Changing RF conditions

`006` is an ideal proton turn map with prescribed transition gamma and native
reference-energy gain. It shows the focusing phase branch on either side,
without claiming an adiabatic capture or real-machine acceleration programme.

`007` uses a 26 GeV/c proton reference, two absolute-time cavities and an RF-free
map with momentum-dependent slip. Each train has three small Gaussian bunches,
linearly matched with only its own RF powered. Native multibunch generation
uses that system's bucket spacing; both RF systems then act on both trains. These isolated matches are not equilibria of the combined time-dependent
potential. No final recapture, acceptance or stacking yield is modelled.

Xpart provides native bucket construction and longitudinal matching. Return to
[RF and longitudinal motion](../README.md) for clocks, acceleration and optics.
