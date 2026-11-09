<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Transverse optics

Read in order; each example runs independently. Periodic and open optics start
this route, followed by inspecting the results and computing beam covariance.

| Example | Lesson |
| --- | --- |
| [001_periodic_twiss.py](001_periodic_twiss.py) | Solve periodic transverse optics and compare boundary conditions and phase advances |
| [002_open_twiss.py](002_open_twiss.py) | Propagate specified entrance beta, alpha and dispersion through a transport line |
| [003_inspect_twiss.py](003_inspect_twiss.py) | Access global results, scalar entries, column arrays and selected rows |
| [004_beam_covariance.py](004_beam_covariance.py) | Convert emittances to covariance, inspect an x-px block and include independent momentum spread |

The [native magnetic models](../../../test_data/example_lattices/README.md) use
1 GeV/c electrons. These calculations are 4D, with no RF, radiation or collective
forces. The ring's optics are periodic; the transport line's entrance conditions
are supplied. `line.twiss4d(...)` is the explicit 4D form of
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

For normalized launches and bunch preparation, see [particles](../../particles/).
For measured beam envelopes with `ONE_TURN_EBE`, see
[element-by-element monitoring](../../monitors/002_element_by_element.py).
For general table operations, see [tables](../../tables/).
