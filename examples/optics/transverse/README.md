<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Transverse optics

Start with periodic or open Twiss, then learn the boundary conditions before
working with covariance, coupling and maps. Each script runs independently.

| Example | Lesson |
| --- | --- |
| [001_periodic_twiss.py](001_periodic_twiss.py) | Solve periodic optics and interpret phase advances and boundary conditions |
| [002_open_twiss.py](002_open_twiss.py) | Propagate specified entrance beta, alpha and dispersion |
| [003_twiss_initial_conditions.py](003_twiss_initial_conditions.py) | Supply complete upstream, downstream or interior conditions, including the particle and W matrix |
| [004_inspect_twiss.py](004_inspect_twiss.py) | Access results, select rows and reuse axes with native Twiss plotting |
| [005_beam_covariance.py](005_beam_covariance.py) | Convert emittances to covariance and include an independent momentum spread |
| [006_coupled_mode_optics.py](006_coupled_mode_optics.py) | Resolve both modes in each physical plane and obtain projected sizes |
| [007_edwards_teng.py](007_edwards_teng.py) | Interpret the local Edwards–Teng basis, propagated functions and decoupled transfer maps |
| [008_periodic_cells.py](008_periodic_cells.py) | Compare a periodic cell, a partial-periodic range and the same cell in a full ring |
| [009_cell_stability_boundary.py](009_cell_stability_boundary.py) | Track cell maps across the linear stability boundary without requesting unstable periodic optics |
| [010_change_starting_point.py](010_change_starting_point.py) | Distinguish periodic and open wrapping, coordinate origins and cycling |
| [011_reverse_reference_frame.py](011_reverse_reference_frame.py) | Compare orbit, optics and strength conventions in opposite local frames |
| [012_closed_orbit_search.py](012_closed_orbit_search.py) | Find and track a transverse fixed point, then reuse it for optics |
| [013_closed_orbit_dispersion.py](013_closed_orbit_dispersion.py) | Differentiate off-momentum closed orbits at fixed reference momentum |
| [014_transfer_matrices.py](014_transfer_matrices.py) | Compose 4D/6D maps and compare transported mode shapes with tracked linearization |
| [015_twiss_settings.py](015_twiss_settings.py) | Choose output content and inspect numerical finite-difference settings |
| [016_save_and_load_twiss.py](016_save_and_load_twiss.py) | Preserve native Twiss snapshots and distinguish file layout from convention conversion |

## Models and coordinates

Most lessons use the [demonstration ring](../../../test_data/example_lattices/demo_ring/README.md):
1 GeV/c electrons, 1.5 km circumference, RF voltage and nonlinear strengths zero,
no radiation or collective force. Open transport lessons retain the
[native transport](../../../test_data/example_lattices/transport/lattice.py).
`twiss4d()` fixes momentum rather than imposing RF phase closure.

Rows refer to element entrances and a final `_end_point`. In a ranged call that
last row is the selected final element's exit, not necessarily the line exit.
Positions, beta functions and position dispersion are in metres; alpha is
dimensionless. `px`/`py` are canonical momenta normalized to reference momentum.
`dpx`/`dpy` differentiate them with respect to `delta=(p-p0)/p0`. Phase advances
are in turns and retain their integer part.

## Covariance and coupling

`005` starts with centered transverse emittances and no longitudinal-mode
population. Its independent momentum-spread contribution is a first-order
uncoupled construction, not an RF-matched bunch. In coupled optics, the
emittances populate modes rather than two independent physical planes.

`006` and `007` add two skew gradients to their own model loads. Native
Edwards–Teng parameters vary with position: a map between two points needs the
local decoupling basis at each end. The propagation check is a consistency
relation among native Twiss/maps, not independent field integration. The chosen
mode branch is valid here; degeneracies need separate treatment.

## Boundary conditions and maps

A complete `TwissInit` carries orbit, optical modes, phases and available
chromatic derivatives. The open examples use zero-length markers to define the
physical boundary planes unambiguously. Periodicity on a selected range solves
that range's optics; it need not close geometrically. The cell's periodic
dispersion can differ from its dispersion when embedded in the ring.

`009` uses the trace criterion only for uncoupled symplectic 2×2 transverse
blocks at fixed momentum. An elliptic linear map is not a nonlinear acceptance
or long-term stability certificate. `010` distinguishes a full periodic turn
from an open range crossing the stored sequence seam. Reversing the frame in
`011` changes table conventions; it does not construct a second beam model.

Transfer maps use `(x,px,y,py,zeta,pzeta)` column vectors. Entry-dependent units
matter, and map endpoints must coincide with the tracking observation planes.
Pairwise W-column outer products remove arbitrary normal-mode phase when
comparing transported mode shapes. Numerical matrix steps require their own
convergence checks for each lattice.

`016` includes HDF5 and TFS formats and therefore needs **h5py** and
**tfs-pandas**, in addition to Xtrack, NumPy and Matplotlib. Native JSON retains
the complete snapshot; a selected TFS layout is not full metadata preservation.

## Continue by subject

[Chromatic optics](../nonlinear/README.md) owns tune/local-optics momentum
response, including higher-order derivatives. [Longitudinal optics](../longitudinal/README.md)
powers RF and solves the synchronous 6D state. For measured beam envelopes see
[element-by-element monitoring](../../monitors/particle_records/002_element_by_element.py).
General table operations live in [tables](../../tables/README.md).
