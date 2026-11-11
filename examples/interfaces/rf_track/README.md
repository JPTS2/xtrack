<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# RF-Track

Use `xt.RFT_Element` to call RF-Track transport from Xtrack. These independent
examples use CPU electron probes with fixed reference momentum.

| Example | Lesson |
| --- | --- |
| [001_track_an_external_element.py](001_track_an_external_element.py) | Wrap an external drift and inspect its agreement with exact native drift transport |
| [002_rf_fieldmap_transport.py](002_rf_fieldmap_transport.py) | Load a complex longitudinal RF map and scan energy gain against entrance arrival time |

The optional backend is [RF-Track](https://gitlab.cern.ch/rf-track), developed by
Andrea Latina at CERN. It is distributed for Python as
[RF-Track on PyPI](https://pypi.org/project/RF-Track/).
These examples were verified with RF-Track 2.7.2 and Xtrack 0.118.7 on macOS ARM64
using CPU execution. RF-Track has its own integration and field-map conventions;
see its [reference manual](https://zenodo.org/records/22086072).

The bridge converts Xtrack metres, normalized canonical transverse momenta,
`zeta` and `delta` into RF-Track's millimetres, milliradian slopes, time in mm/c
and momentum in MeV/c. The probes have zero transverse vector potential at the
interface, so canonical and mechanical transverse momenta coincide. For an
incoming particle,
`P_s/p0 = sqrt((1+delta)**2 - px**2 - py**2)` and the slopes are `P_x/P_s` and
`P_y/P_s`. The entrance time is `t = -zeta/(beta0*c)`: positive `zeta` arrives
earlier. `001` checks finite-angle and off-momentum probes, including their
longitudinal transport.

`002` reads the existing [CLIC complex on-axis map](../../rf_track/400_Ez_on-axis_TD26__vg1p8_r05_CC_EDMS.txt).
Its header gives longitudinal positions in millimetres and a frequency of
11.994 GHz, but does not establish field units or an input-power calibration.
The example keeps the relative complex field shape and scales its maximum
magnitude to the explicit `PEAK_EZ` parameter. The plotted gain therefore belongs
to that illustrative field, rather than a calibrated CLIC operating point.
Both real and imaginary parts are retained.

The RF clock origin and additional phase are set explicitly. The scan variable
is entrance arrival-time phase, `360*f*t`; zero need not be the accelerating
crest. `update_ref=False` keeps `p0c` unchanged, so acceleration appears in
`delta`, `ptau` and the public `particles.energy` property. The sample passes
through one external element with an element-local clock.

These examples use homogeneous, initially live CPU particles. General
pre-existing loss handling, mixed-species transport, accelerating-reference
updates and repeated-element absolute RF timing are outside the verified bridge
scope. For native synchronized RF clocks and acceleration, see
[RF timing](../../rf/timing/README.md) and
[acceleration](../../rf/acceleration/README.md).

The older [RF-Track examples](../../rf_track/) retain magnetic volumes,
solenoids, loss and ring-comparison workflows.
