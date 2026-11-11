<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# LHC collision optics and orbit knobs

Load a native HL-LHC design snapshot, inspect its collision optics, then apply
its installed IP1 crossing and separation knobs. The example measures the
resulting orbit rather than assuming exact agreement with the requested values.

| Example | Lesson |
| --- | --- |
| [001_collision_optics_and_orbit_knobs.py](001_collision_optics_and_orbit_knobs.py) | Inspect interaction-point optics, apply installed crossing and separation bumps, and measure their local response and residual orbit elsewhere |

The [native model](../../../test_data/hllhc15_thick/lhc_thick_with_knobs.json)
contains beam 1 with a 7 TeV/c proton reference. Its nominal full tunes are
62.31 and 60.32; round beta functions are 0.15 m at IP1/IP5, 1.5 m at IP8 and
10 m at IP2. The script computes 4D optics at `delta = 0`. It uses Xtrack,
NumPy and Matplotlib; loading this JSON requires no MAD-X installation.
This is a stored design model, not a claim about current machine settings.

With the stored `phi_ir1 = 0`, `on_x1` controls horizontal crossing in microradians,
and `on_sep1` controls vertical displacement in millimetres. Positive values
request positive beam-1 `px` and `y`, respectively. Here `px = P_x/p0`; at the
reference momentum it is the small-angle trajectory slope. A 150 microradian
setting is the requested angle of this one beam, and a 0.5 mm setting is its
offset. Neither is a directly simulated two-beam crossing angle or separation.
The script does not track beam 2 or compute luminosity or beam-beam effects.

The script sets `on_disp = 0` to isolate the orbit bumps from the stored
companion dispersion-correction knobs. Keeping the snapshot's `on_disp = 1`
also excites orbit outside IR1; that combined correction is a different task.
The orbit plot covers 600 m on either side of IP1, including the insertion
boundaries `s.ds.l1.b1` and `e.ds.r1.b1` at about -547 m and +547 m.
The terminal reports maximum orbit changes outside those boundaries.
Applying these orbit knobs does not rematch the optics or certify aperture
clearance.

The [model builder](../../../test_data/hllhc15_thick/000_build_thick_line.py)
records how the native snapshot was produced from the legacy sequence and
strength files. Its thick optics and crossing response are exercised in
[test_thick_lhc.py](../../../tests/test_thick_lhc.py).
For constructing new local bumps and matching interaction-region optics, see
the [matching examples](../../matching/) and the more extensive two-beam
regressions in
[test_match_optics_and_ip_knob.py](../../../tests/test_match_optics_and_ip_knob.py).
