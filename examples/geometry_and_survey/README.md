<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Geometry and survey

Read in order; each script runs independently. Survey describes reference
geometry, rather than a tracked orbit. Begin with an open line, check a ring's
closure, place it in a chosen frame, then inspect actual misaligned elements.
Plots use Matplotlib. `001` also needs Xplt: `survey.plot` delegates the floor
plot to `xplt.FloorPlot`.

| Example | Question |
| --- | --- |
| [001_survey_a_line.py](001_survey_a_line.py) | Read positions and angles and draw the native transport line |
| [002_ring_closure.py](002_ring_closure.py) | Check ring closure in position and orientation |
| [003_survey_from_a_selected_point.py](003_survey_from_a_selected_point.py) | Assign a global pose and reconstruct the same local closed orbit in two placements |
| [004_element_frames.py](004_element_frames.py) | Compare reference/actual frames, preserve thick-slice endpoints and save a survey snapshot |
| [005_nonplanar_coordinate_bump.py](005_nonplanar_coordinate_bump.py) | Reconstruct a straight physical ray through nonplanar reference-frame changes |
| [006_rectangular_bend_frames.py](006_rectangular_bend_frames.py) | Relate horizontal/vertical rectangular-bend faces, sliced reference paths and physical rays |

`001` uses the native transport line; `002` and `003` use the
[shared demo ring](../../test_data/example_lattices/demo_ring/README.md).
Default survey axes are global `(X, Y, Z)`, initially horizontal, vertical and
forward. Positions are in metres and angles are in radians. `s` is accumulated
reference-path length. It is distinct from global `Z`. Orientation closure
is checked with frame matrices: an unwrapped angle changing by a full turn is
consistent with the same final orientation. The polylines in `002` and `003`
join element boundaries with straight chords; they do not resolve curved bend
bodies.

`003` supplies a pose at `bpm.s4.f` using `element0`. All rows, including those
upstream of that entrance, remain in their original line order. The example's
horizontal plot is a projection if you change the elevation or roll.
Its orientation angles are unwrapped along the original line, so the selected
entrance's theta can differ from its assigned value by a multiple of `2*pi`.
The 4D closed orbit has fixed `delta = 1e-4`, with RF and radiation off. Its
positions are reconstructed on the surveyed transverse planes using the local
`ex` and `ey` axes. The floor plot magnifies only the orbit displacement by the
labelled factor; a separate plot shows the physical displacement in millimetres.
Changing the global pose rotates/translates the result without changing that
local orbit.

`004` uses a small inline quadrupole line. `rot_shift_anchor` is a distance in
metres from the element entrance. It compares the reference and actual element
frames, transforms the same local point to global coordinates, then expresses
the actual frame in CCS conventions. CCS conversion changes the coordinate
basis; it does not locate the model geographically at CERN. Its azimuth is
reported in gon while the other survey angles retain radians.
Thick slices of a separate copy preserve this quadrupole's physical entrance
and exit frames. The native JSON snapshot retains the surveyed frame arrays,
without embedding a live line; save the model separately for new calculations.
The example writes `outputs/element_frames.json` beside the script.

CCS basis conversion is distinct from interpreting GEODE displacement reports.
It does not choose radial/RST displacement signs, a socket pivot or a policy for
missing endpoint measurements. The retained [machine survey workflow](../survey_ccs/)
still uses a private geodesy helper and machine inputs; those contracts must be
settled before it becomes a recommended native example.

For placement and shared objects, see [line construction](../line_construction/).
For particle coordinate transformations, see
[spatial transformations](../elements/transformations/README.md).
Signed open geometries and machine-specific alignment workflows remain in the
[survey examples](../survey/).

## Nonplanar coordinates and rectangular magnets

`005` interleaves yaw, pitch and roll changes. Local transverse coordinates
make excursions while their global reconstruction stays on one field-free
straight ray. A passive change of coordinate plane is not a magnetic kick or
a physical closed-orbit bump. Global Z and the reference bookkeeping s need not
coincide at the differently oriented planes.

`006` rotates the same rectangular magnet into horizontal and vertical bending
planes, then compares straight-body and curved-body representations with thick
slices. The field setting and chord/angle are fixed. Straight-body face maps
carry the reference-angle changes at their boundaries; curved-body slices
carry distributed curvature. Local coordinates can differ despite the same
global exit ray. Tables and plots expose this geometry; the comparison is not
a proof that all finite-amplitude maps or coarse slice counts are equivalent.
