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
| [006_solenoids.py](006_solenoids.py) | Helical motion, on-axis spin compensation and a tabulated detector field |
| [007_spline_fields.py](007_spline_fields.py) | Supply derivative profiles, propagate open optics and compare a rotated uniform field |
| [008_field_expansion.py](008_field_expansion.py) | Combine polynomial fields, live strengths and scaling; retain parents through slicing |
| [009_elliptic_nonlinear_lens.py](009_elliptic_nonlinear_lens.py) | Compare exact kicks and see how an elliptic lens bends a phase-space grid |
| [010_callable_fieldmap.py](010_callable_fieldmap.py) | Track through a solenoid field defined as a Python function |
| [011_smooth_dipole_fringes.py](011_smooth_dipole_fringes.py) | Compare hard edges and smooth on-axis dipole ramps at equal integral and exterior planes |

`001` uses a positive-charge proton reference at 1 GeV/c. Physical
fields are normalized to its signed rigidity. Distributed normal strengths
`k_n` have units m^(-n-1); `knl[n]` is their longitudinal integral, in m^(-n).
The multipole polynomial includes `1/n!`; the stored coefficient does not.
The thin-kick comparison uses the reference species at `delta = 0` and `y = 0`.
The separate `009` lesson uses `NonLinearLens` within its regular domain and
compares the finite-amplitude horizontal kick with the exact axis derivative.
A tracked horizontal phase-space grid shows where the kick departs from its
linear quadrupole approximation.

`002` uses the [shared demo ring](../../../test_data/example_lattices/demo_ring/README.md)
with its 1 GeV/c electron reference and 4D periodic optics. It changes body
settings while retaining strengths, bend edges and reference geometry. The
resulting tune differences describe those settings, rather than rank accuracy.
Its focusing-family regex selects arc QF halves; the insertion quadrupoles
retain the class-level settings. RF voltage is zero and radiation is off.

`003` first compares one fringe-free quadrupole with a common 1 GeV/c proton
ensemble. The thin and thick representations preserve geometric
length and integrated strength. A bounded scan from 1 to 256 slices compares
their exit maps with the unsliced adaptive parent, keeping the entrance probes
and physical fields fixed. Thin-map errors decrease; thick slices agree to
round-off for this fringe-free quadrupole. The plot uses Matplotlib; `px` and
`py` are normalized canonical momenta. Larger numerical studies belong in the
convergence benchmarks, rather than this introductory slicing lesson.
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

`006` follows two protons through a main solenoid and a compensator of equal
length and opposite polarity. The tilted probe traces a helix in each magnet.
The second proton stays on axis, where the signed field integrals determine
the net spin phase. The compensator unwinds that phase while the tilted probe
continues to move transversely. Thick slices sample the motion inside each
magnet and retain its exterior hard edges. A short launch on a displaced
solenoid axis shows that `x0` leaves an embedded quadrupole in place, whereas
`shift_x` moves both fields.

The second part of `006` builds a piecewise-linear `VariableSolenoid` profile from the FCC-ee
detector and screening field table. Adjacent segments share endpoint strengths.
Each segment uses the mean strength for the orbital map. The local example
aligns the beam and detector axes and closes the field at the two end planes.
See the [FCC-ee insertion study](../../machine_studies/fcc_ee/detector_solenoid/)
for the tilted detector and matching in a ring. Example `010` introduces the
spatial Boris interface for a field defined by a function.

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

## Analytic fields and dipole profiles

`010` supplies a Python function that returns `(Bx, By, Bz)` in tesla from
Cartesian positions in metres. The function evaluates a near-axis
solenoid profile with divergence-free linear transverse terms. The spatial
Boris coordinates are Cartesian and its momenta are kinetic. Zero endpoint
fields make them agree with exterior canonical momenta. A sampled native
`VariableSolenoid` line supplies an independent finite representation. Doubling
Boris steps checks that discretization alone; it does not certify an exact
finite-coil field or the sampled profile's convergence.

`011` compares two quintic `BFieldExpansion` fringes with full `Bend` edge
maps. It includes a zero-`hgap`, zero-`fint` Bend baseline and derives
`hgap * fint` from each resolved profile for the paired comparisons.
All cases use a straight reference, the same peak field and field integral,
and common field-free end planes. The script checks RK4 refinement, then
subtracts the small-amplitude slope of the exit-momentum difference. It shows
the total kicks alongside the nonlinear residual as a fraction of the
corresponding matched kick.
