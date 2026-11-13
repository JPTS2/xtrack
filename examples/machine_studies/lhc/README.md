<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# LHC collision optics with shared magnets

Use one native two-beam HL-LHC design snapshot to connect installed collision
knobs, counter-rotating reference frames and a joint optics match. Each script
loads the snapshot independently; no generated files or MAD-X installation are
required. Xtrack, NumPy and Matplotlib are used.

| Example | Lesson |
| --- | --- |
| [001_two_beam_geometry_and_shared_magnets.py](001_two_beam_geometry_and_shared_magnets.py) | Put the two surveys in one global frame and observe both signed element representations respond to a common triplet circuit |
| [002_collision_optics_and_orbit_knobs.py](002_collision_optics_and_orbit_knobs.py) | Apply installed beam1 crossing/separation knobs and measure local response and residual orbit elsewhere |
| [003_match_two_beam_collision_optics.py](003_match_two_beam_collision_optics.py) | Change the IP8 beta functions and restore collision waists and tunes in one two-beam optimization |

The [native model](../../../test_data/hllhc15_thick/hllhc15_collider_thick.json)
contains 7 TeV/c proton references. Nominal full tunes are 62.31/60.32, with round
beta functions 0.15 m at IP1/IP5, 1.5 m at IP8 and 10 m at IP2. This is a stored
design model, not a statement about current operational settings.

## Knob and reference conventions

With stored `phi_ir1 = 0`, `on_x1` requests horizontal beam1 crossing in
microradians and `on_sep1` requests vertical displacement in millimetres.
Positive settings request positive beam1 `px` and `y`. `px = P_x/p0` is the
small-angle trajectory slope at the reference momentum. Neither setting alone
is a two-beam crossing angle or separation. The collision-knob lesson sets
`on_disp = 0` to isolate these orbit bumps from their companion dispersion-correction knobs.

Beam2 is represented in its own tracking direction, with a saved reversed-Twiss
convention. The survey example supplies opposite IP1 tangents to express the
two reference trajectories in one global frame. Shared magnets have separate
beam-element representations; a common circuit controls their signed fields.
Sharing circuit values is a physical constraint, rather than a reason to match
the beams in separate environments.

## Collision matching scope

The joint match increases both IP8 beta functions from 1.5 to 1.65 m, enforces
zero alpha and preserves both tunes. Shared triplet circuits and independent
matching-section quadrupoles are varied together. Independent periodic Twiss
results show the other IPs, dispersion and resulting chromaticities. These
quantities are not constrained by the selected targets. The calculation does
not preserve every insertion boundary, impose hardware limits, certify aperture
clearance or include beam-beam interactions; it is not an operational squeeze.

The [model builder](../../../test_data/hllhc15_thick/001_build_thick_collider.py)
records the reference-direction conventions. For optimizer tools and generic shared-element matching, see
[matching](../../matching/).
