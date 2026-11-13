<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Spatial transformations

Use [001_translations_and_rotations.py](001_translations_and_rotations.py) to
inspect reference-frame changes and their relation to magnet misalignment.
It uses a field-free proton launch with a 1 GeV/c reference and `delta = 0.01`,
keeping the reference momentum fixed. Distances are in metres; `Rotation` takes angles in radians.

The example follows three steps:

1. **Transverse signs.** A positive `Translation.shift_x` gives
   `x_new = x_old - shift_x`. A roll gives
   `x_new = cos(theta)*x + sin(theta)*y` and
   `y_new = -sin(theta)*x + cos(theta)*y`, with the same rotation of `px, py`.
   These express coordinates in the changed frame.
2. **A displaced and tilted plane.** The 3D example uses a longitudinal shift
   and `Rotation(seq = 'yxs')`, applying y, then x, then s rotations.
   A longitudinal translation propagates the ray
   to the displaced plane while keeping the bookkeeping coordinate `s` fixed;
   `x`, `y` and `zeta` can change. Pitch and yaw also change the observation
   plane. Inverse tracking reverses both the element order and the order/signs
   of the rotations, returning to the original coordinates.
3. **A magnet in that framework.** For transverse shift and roll, an explicit
   transform–quadrupole–inverse-transform sequence reproduces the quadrupole's
   `shift_x`, `shift_y` and `rot_s_rad` attributes. The displayed magnet exits
   are all in the original frame, so they can also be compared with an aligned
   quadrupole. A frame change by itself applies no magnetic force; the magnet
   between the transforms changes the physical ray.

The magnet comparison uses transverse offsets and roll only; it does not extend
that equivalence to arbitrary pitch/yaw anchors. The quadrupole has no fringe
fields in this example. The final tables separate coordinate-frame results from
magnet exits, and label positions in millimetres and `s` in metres. `px` and
`py` are dimensionless canonical momenta normalized to reference momentum.

For model placement, see [line construction](../../line_construction/README.md).
Reference paths, actual element frames and nonplanar reconstructions are
covered by [geometry and survey](../../geometry_and_survey/README.md).
More extensive alignment applications remain under
[element transformations](../../element_transformations/) and
[misalignments](../../misalignments/), with their own inputs and requirements.
