<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Transverse optics

Read in order; each example runs independently. Periodic and open optics start
this route, followed by inspecting results, beam covariance and coupled optics.

| Example | Lesson |
| --- | --- |
| [001_periodic_twiss.py](001_periodic_twiss.py) | Solve periodic transverse optics and compare boundary conditions and phase advances |
| [002_open_twiss.py](002_open_twiss.py) | Propagate specified entrance beta, alpha and dispersion through a transport line |
| [003_inspect_twiss.py](003_inspect_twiss.py) | Access global results, scalar entries, column arrays and selected rows |
| [004_beam_covariance.py](004_beam_covariance.py) | Convert emittances to covariance, inspect an x-px block and include independent momentum spread |
| [005_coupled_mode_optics.py](005_coupled_mode_optics.py) | Resolve both normal modes in each plane and combine mode emittances into projected sizes |
| [006_edwards_teng.py](006_edwards_teng.py) | Read Edwards-Teng optics and use the coupling matrix to decouple the transverse one-turn map |
| [007_twiss_initial_conditions.py](007_twiss_initial_conditions.py) | Reuse a complete Twiss state at upstream or downstream boundaries of a subrange |
| [008_periodic_cells.py](008_periodic_cells.py) | Solve a repeated cell and compare its optics and phase advances with the full ring |
| [009_change_starting_point.py](009_change_starting_point.py) | Distinguish a wrapped calculation, coordinate zeros and cycling the line |
| [010_reverse_reference_frame.py](010_reverse_reference_frame.py) | Transform orbit and optical quantities into the opposite local reference frame |
| [011_closed_orbit_search.py](011_closed_orbit_search.py) | Find and track a transverse fixed point, reuse it for optics and choose a search location |
| [012_closed_orbit_dispersion.py](012_closed_orbit_dispersion.py) | Differentiate off-momentum closed orbits and compare their shifts with dispersion |
| [013_transfer_matrices.py](013_transfer_matrices.py) | Compose entrance-to-entrance maps and compare Twiss, finite differences and displaced-particle tracking |
| [014_twiss_settings.py](014_twiss_settings.py) | Choose result content and line defaults, then check finite-difference step sensitivity |
| [015_save_and_load_twiss.py](015_save_and_load_twiss.py) | Save a complete Twiss result and continue analysis after loading it |

Most lessons use the [shared demonstration ring](../../../test_data/example_lattices/demo_ring/README.md)
with its 1 GeV/c electron reference and 1.5 km circumference. RF voltage and
nonlinear strengths are zero; radiation is off and no collective force is
included in these 4D calculations. The open-line lessons `002`, `007` and `013`
retain the [native transport line](../../../test_data/example_lattices/transport/lattice.py):
they supply entrance conditions rather than imposing ring periodicity.
`line.twiss4d(...)` is the explicit 4D form of
`line.twiss(method='4d', ...)`.

Twiss rows describe element entrances and a final exit called `_end_point`.
Beta and position dispersion are in metres; alpha is dimensionless. Phase
advances `mux` and `muy` are in turns, without reduction modulo one. `dpx` and
`dpy` differentiate normalized canonical momenta with respect to `delta`.
The reference-species convention is `delta = (p-p0)/p0`.

`004` uses transverse rms emittances about the centroid. Its covariance axes are
`(x, px, y, py, zeta, pzeta)`; the displayed block is `(x, px)`, not `(x, x')`.
The uncoupled model lets us compare the API with the Courant-Snyder formula.
For an independently imposed momentum spread, the example explicitly adds the
first-order dispersion contribution to the transverse covariance. This is not
an RF-matched longitudinal distribution. Coupled and longitudinal covariance
need their corresponding optical modes and assumptions.

`005` and `006` power `ks_s2` and `ks_s5` in independent loads of DEMO.
These add skew components of +0.001 and -0.0008 m^-2 to the 1 m quadrupoles
`arc.s2.c2.qd` and `arc.s5.c2.qd`. The supplied strengths give stable, distinct
transverse modes with nonzero cross-plane projections. No skew fields are added to the
model file. Mode labels follow the horizontal/vertical branches near the
uncoupled limit; they are not two independent physical-plane oscillations.
The covariance API's `nemitt_x/y` specify the two mode populations here.

The Edwards-Teng example uses canonical transverse coordinates and a forward
periodic 4D solution. Its `r*_edw_teng` entries form a coupling matrix, distinct
from the transport `R_matrix`. The example states its transformation convention
and shows the resulting one-turn blocks. This parameterization needs a suitable
mode branch; it is not a universal coordinate chart at mode degeneracies.
Constructing coupling bumps and knobs belongs with matching.

`007` propagates a complete `TwissInit`, including orbit, dispersion and phase
origins. A zero-length marker defines its downstream boundary. A ranged
calculation includes its final element; its `_end_point` is
that element's exit, which need not be the full line's final exit.

`008` solves the reusable `arc.cell`, with a 15-degree geometric bend and
90-degree betatron phase advance in each plane. The straight sections match
its periodic beta/alpha, but the assembled ring has zero dispersion at the
arc ports; that differs from the periodic cell's dispersion. Cell phase alone
does not give the ring tunes, because the straight sections add phase. Nor does
the four-cell transverse identity determine a unique periodic beta solution.
Periodicity is an optical boundary condition; the cell need not close on the
floor plan.

`009` separates the calculation's starting point from the stored line order and
the zeros of `s` and phase advance. `010` changes the local reference frame:
reversed entrances correspond to proper-frame exits. Its sign comparisons use
zero-length markers to identify the same physical locations. Small quadrupole
offsets make both orbit planes visible; these offsets exist only in that load.

`011` searches for transverse periodicity at fixed `delta` in a displaced
quadrupole lattice. Without RF it does not require longitudinal phase closure.
The returned particle is tracked for one turn, then reused for optics.
`co_search_at` moves the search location while keeping the output line order.

`012` keeps the reference momentum and magnetic strengths fixed.
Dispersion differentiates the closed orbit; `dpx/dpy` refer to normalized
canonical momenta. A finite-delta orbit also contains higher-order terms.
The centered difference uses delta = +/-1e-5. The imposed spread in `004`
is also small: these examples demonstrate local derivatives and linear covariance,
not a qualified momentum acceptance. DEMO's bare chromaticity is large and its
vertical tune is close to an integer.
Tune and local-optics momentum dependence belong together in the
[chromatic optics sequence](../../README.md#nonlinear), within nonlinear optics.
Matching chromaticity belongs with matching.

`013` uses canonical column vectors `(x, px, y, py, zeta, pzeta)` and compares
the fixed-momentum transverse block. Map endpoints are element entrances;
tracking excludes its stop element to use that same plane. Predictions describe
displacements from the reference trajectory, with nonlinear and numerical
residuals. Transfer-matrix entries carry coordinate-dependent units.

`014` sets defaults only on its freshly loaded line and compares requested
outputs and finite-difference steps. Its numerical check is qualified for this
smooth model; step-size convergence depends on the lattice and amplitudes.

`015` writes `outputs/twiss.json` beside the script. The result includes optical
matrices, phase advances and reference-particle data for subsequent analysis.
It is distinct from saving the associated line or environment, covered in
[environment persistence](../../line_construction/015_save_and_load_environment.py).
Other table formats and column selection are covered in [tables](../../tables/).

For normalized launches and bunch preparation, see [particles](../../particles/).
For measured beam envelopes with `ONE_TURN_EBE`, see
[element-by-element monitoring](../../monitors/particle_records/002_element_by_element.py).
For general table operations, see [tables](../../tables/).
