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
| [003_survey_from_a_selected_point.py](003_survey_from_a_selected_point.py) | Assign global coordinates and angles at a chosen entrance without cycling the line |
| [004_element_frames.py](004_element_frames.py) | Compare reference and actual element frames and transform local coordinates |

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

`004` uses a small inline quadrupole line. `rot_shift_anchor` is a distance in
metres from the element entrance. It compares the reference and actual element
frames, transforms the same local point to global coordinates, then expresses
the actual frame in CCS conventions. CCS conversion changes the coordinate
basis; it does not locate the model geographically at CERN. Its azimuth is
reported in gon while the other survey angles retain radians.

For placement and shared objects, see [line construction](../line_construction/).
For particle coordinate transformations, see
[spatial transformations](../elements/transformations/README.md).
Signed open geometries and machine-specific alignment workflows remain in the
[survey examples](../survey/).
