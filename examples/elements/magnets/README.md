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
| [001_fields_and_strengths.py](001_fields_and_strengths.py) | Reference rigidity, distributed strengths and integrated multipole conventions |
| [002_models_and_integrators.py](002_models_and_integrators.py) | Query model choices and apply class, family and individual settings |
| [003_thin_and_thick_slices.py](003_thin_and_thick_slices.py) | Choose slicing rules and occurrences; inspect inherited strengths/alignment and exit maps |
| [004_bend_geometry_and_edges.py](004_bend_geometry_and_edges.py) | Separate bend geometry, powered fields, edge maps and rectangular body models |
| [005_fringe_fields_and_drifts.py](005_fringe_fields_and_drifts.py) | Compare hard-edge settings and a resolved smooth fringe at common exterior planes |
| [006_solenoids.py](006_solenoids.py) | Uniform/ramped solenoids, equal-integral comparison and field-axis offsets |
| [007_spline_fields.py](007_spline_fields.py) | Supply derivative profiles, propagate open optics and compare a rotated uniform field |
| [008_field_expansion.py](008_field_expansion.py) | Combine polynomial fields, live strengths and scaling; retain parents through slicing |
| [009_elliptic_nonlinear_lens.py](009_elliptic_nonlinear_lens.py) | Compare a non-polynomial elliptic lens with its exact axis derivative and linear limit |
| [010_callable_magnetic_field.py](010_callable_magnetic_field.py) | Supply a physical three-component field to the public spatial Boris interface |
| [011_smooth_dipole_fringes.py](011_smooth_dipole_fringes.py) | Compare hard edges and smooth on-axis dipole ramps at equal integral and exterior planes |
| [012_solenoid_field_axes.py](012_solenoid_field_axes.py) | Distinguish solenoid/embedded-multipole axes and check translated variable-field radiation transport |

`001` uses a positive-charge proton reference with `p0c = 1e9` eV. Physical
fields are normalized to its signed rigidity. Distributed normal strengths
`k_n` have units m^(-n-1); `knl[n]` is their longitudinal integral, in m^(-n).
The multipole polynomial includes `1/n!`; the stored coefficient does not.
The thin-kick comparison uses the reference species at `delta = 0` and `y = 0`.
The separate `009` lesson uses `NonLinearLens` within its regular domain and
compares the finite-amplitude horizontal kick with the exact axis derivative.
It is a non-polynomial potential rather than a finite multipole expansion.

`002` uses the [shared demo ring](../../../test_data/example_lattices/demo_ring/README.md)
with its 1 GeV/c electron reference and 4D periodic optics. It changes body
settings while retaining strengths, bend edges and reference geometry. The
resulting tune differences describe those settings, rather than rank accuracy.
Its focusing-family regex selects arc QF halves; the insertion quadrupoles
retain the class-level settings. RF voltage is zero and radiation is off.

`003` first compares one fringe-free quadrupole with a common 1 GeV/c proton
ensemble. The thin and thick representations preserve geometric
length and integrated strength. Their exit maps are compared with the unsliced
adaptive parent at one chosen slice count. Numerical convergence needs its own
study. The plot uses Matplotlib; `px` and `py` are normalized canonical momenta.
Later sections show class/name rules, last-match precedence, a no-slice override
and one repeated occurrence. Tables expose parent-linked normal/skew strengths
and alignment; absolute error strengths are kept separate from relative errors.

`004` uses a curved-body `RBend` and a sector `Bend` with the same reference arc
and angle. Changing angle at fixed chord also changes arc length and the
field-following `k0`. Its tracking comparison uses the original geometry, linear
edge maps and zero fringe integrals. Zero user edge angles mean different face
incidences for these two bend types. Later sections hold a powered `k0` fixed
while changing reference geometry, restore bend definitions and compare curved-
and straight-body rectangular models using full edges and common exit frames.
These endpoint comparisons do not replace a full trajectory/vertical-geometry
study.

`005` holds the quadrupole body map fixed while changing its hard-edge fringe
maps or the models of the two surrounding drifts. Exit differences use the
fringe-on/exact-drift case as a stated comparison baseline. A separate comparison
resolves smooth C2 quadrupole ramps, holding the integrated gradient and exterior
planes fixed. Step refinement separates integration error from the finite-profile
effect; equal integrals do not imply identical focusing maps.

`006` uses `UniformSolenoid` with hard-edge entrance/exit maps, then constructs
a continuous on-axis longitudinal-field profile with `VariableSolenoid`. The
uniform and ramped cases have the same peak field but different total lengths and field
integrals; their different exit coordinates describe different magnets. Both
use exterior field-free launch/readout planes. Linear-ramp slopes and the
associated transverse fields jump at ramp boundaries and body joins. Reversing
the uniform field illustrates the sign of the cross-plane response. An additional
sharp-field model matches the ramped integral and exterior planes. For a pure
uniform solenoid, field-axis offsets `x0/y0` reproduce a whole-element shift;
this equivalence is not asserted for embedded multipoles.

`007` and `008` describe the same compact on-axis vertical field, with zero
endpoint values and derivatives, for a 1 GeV/c proton. They run independently.
`Spline4.mean` is a mean field, not an integral. `SplineBoris` takes physical
fields in tesla; `BFieldExpansion.knc` uses rigidity-normalized coefficients in
ascending longitudinal powers. Their field plots convert to tesla consistently.

`008` explicitly chooses `pkin_const = True` for kinetic-momentum boundary
conversions and reports the exit transverse vector potential. Its RK4 stepper
is not exactly symplectic. The initial field comparisons use a straight reference frame,
`y = 0` and no radiation or spin. Their printed momentum integrals provide a
reference for these chosen settings; varying integration and expansion settings
belongs in a convergence study.

The advanced `007` sections add a transverse derivative profile, propagate open
4D optics, and compare uniform rolled fields with an edge-free native `RBend`.
The common reference is straight while the physical field is nonzero.

The advanced `008` sections separate curved reference geometry from physical
field coefficients, combine scalar/integrated strengths and a common scale,
update expressions without reshaping allocated arrays, and inspect parent-linked
thick slices. The automatic expansion order is fixed by the allocated shapes;
changing current coefficient values does not resize it.

For positioning and cutting a line, see
[line construction](../../line_construction/README.md). For slicing strategies, continue with `003` above. For integration studies, see
[convergence benchmarks](../../benchmarks/convergence/README.md). Additional
specialist comparisons remain under
[magnet models and integrators](../../magnet_models_and_integrators/).

Retained advanced studies are available under [solenoids](../../solenoid/),
[spline fields](../../splineboris/) and
[field expansions](../../bfieldexpansion/). These include machine applications,
field fitting and numerical comparisons. Consult their local inputs and
requirements before running them.

## Callable fields, dipole profiles and solenoid axes

`010` supplies a vectorized `(Bx, By, Bz)` callable in tesla, using a near-axis
solenoid profile with divergence-free linear transverse terms. The spatial
Boris coordinates are Cartesian and its momenta are kinetic. Zero endpoint
fields make them agree with exterior canonical momenta. A sampled native
`VariableSolenoid` line supplies an independent finite representation. Doubling
Boris steps checks that discretization alone; it does not certify an exact
finite-coil field or the sampled profile's convergence.

`011` uses a straight reference for both powered dipole models. C2 quintic
ramps shorten the plateau so the on-axis field integral and the field-free
exterior planes match the hard-edge case. A small vertical scan shows different
fringe focusing. The profile is chosen analytically, not fitted to a measured
magnet; ramp-step refinement is distinct from changing the physical profile.

`012` keeps an embedded quadrupole fixed while moving only the solenoid axis,
then moves the whole combined element. These are different fields. A separate
variable-field comparison translates both the ray and field axis and checks
mean-radiation energy loss at equal relative coordinates. It is open transport,
without RF compensation or an equilibrium claim. Neither lesson needs a private
field-fitting or finite-coil helper.
