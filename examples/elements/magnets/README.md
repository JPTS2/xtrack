<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Magnets

Start with physical fields and normalized strengths, then choose a body-map
formulation, an integrator and a thin or thick representation. Each script runs
independently.

| Example | Lesson |
| --- | --- |
| [001_fields_and_strengths.py](001_fields_and_strengths.py) | Reference rigidity, distributed and integrated strengths, and the normal-multipole kick convention |
| [002_models_and_integrators.py](002_models_and_integrators.py) | Query model choices and apply class, family and individual settings |
| [003_thin_and_thick_slices.py](003_thin_and_thick_slices.py) | Inspect slice geometry, parent ownership and strength; compare exit coordinates |
| [004_bend_geometry_and_edges.py](004_bend_geometry_and_edges.py) | Sector versus rectangular geometry, fixed-chord changes and edge focusing |
| [005_fringe_fields_and_drifts.py](005_fringe_fields_and_drifts.py) | Change quadrupole fringe and surrounding drift settings independently |
| [006_solenoids.py](006_solenoids.py) | Uniform and ramped solenoids, field reversal and transverse coupling |
| [007_spline_fields.py](007_spline_fields.py) | Define a physical field with Hermite boundary data and track with SplineBoris |
| [008_field_expansion.py](008_field_expansion.py) | Define normalized polynomial fields and inspect BFieldExpansion tracking |

`001` uses a positive-charge proton reference with `p0c = 1e9` eV. Physical
fields are normalized to its signed rigidity. Distributed normal strengths
`k_n` have units m^(-n-1); `knl[n]` is their longitudinal integral, in m^(-n).
The multipole polynomial includes `1/n!`; the stored coefficient does not.
The thin-kick comparison uses the reference species at `delta = 0` and `y = 0`.

`002` uses the [native magnetic ring](../../../test_data/example_lattices/README.md)
with its 1 GeV/c electron reference and 4D periodic optics. It changes body
settings while retaining strengths, bend edges and reference geometry. The
resulting tune differences describe those settings, rather than rank accuracy.

`003` uses one quadrupole, with fringe fields disabled, and a common proton
input ensemble at 1 GeV/c. The thin and thick representations preserve geometric
length and integrated strength. Their exit maps are compared with the unsliced
adaptive parent at one chosen slice count. Numerical convergence needs its own
study. The plot uses Matplotlib; `px` and `py` are normalized canonical momenta.

`004` uses a curved-body `RBend` and a sector `Bend` with the same reference arc
and angle. Changing angle at fixed chord also changes arc length and the
field-following `k0`. Its tracking comparison uses the original geometry, linear
edge maps and zero fringe integrals. Zero user edge angles mean different face
incidences for these two bend types.

`005` holds the quadrupole body map fixed while changing its hard-edge fringe
maps or the models of the two surrounding drifts. Exit differences use the
fringe-on/exact-drift case as a stated comparison baseline.

`006` uses `UniformSolenoid` with hard-edge entrance/exit maps, then constructs
a continuous on-axis longitudinal-field profile with `VariableSolenoid`. The
uniform and ramped cases have the same peak field but different total lengths and field
integrals; their different exit coordinates describe different magnets. Both
use exterior field-free launch/readout planes. Linear-ramp slopes and the
associated transverse fields jump at ramp boundaries and body joins. Reversing
the uniform field illustrates the sign of the cross-plane response.

`007` and `008` describe the same compact on-axis vertical field, with zero
endpoint values and derivatives, for a 1 GeV/c proton. They run independently.
`Spline4.mean` is a mean field, not an integral. `SplineBoris` takes physical
fields in tesla; `BFieldExpansion.knc` uses rigidity-normalized coefficients in
ascending longitudinal powers. Their field plots convert to tesla consistently.

`008` explicitly chooses `pkin_const = True` for kinetic-momentum boundary
conversions and reports the exit transverse vector potential. Its RK4 stepper
is not exactly symplectic. Both field examples use a straight reference frame,
`y = 0` and no radiation or spin. Their printed momentum integrals provide a
reference for these chosen settings; varying integration and expansion settings
belongs in a convergence study.

For positioning and cutting a line, see
[line construction](../../line_construction/README.md). For slicing strategies,
see the [slicing examples](../../slicing/). Existing longer numerical studies
remain under [magnet models and integrators](../../magnet_models_and_integrators/).

Retained advanced studies are available under [solenoids](../../solenoid/),
[spline fields](../../splineboris/) and
[field expansions](../../bfieldexpansion/). These include machine applications,
field fitting and numerical comparisons. Consult their local inputs and
requirements before running them.
