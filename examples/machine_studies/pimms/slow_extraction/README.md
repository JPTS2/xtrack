<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# PIMMS: resonant slow extraction

Use a compact 200 MeV proton model to connect extraction optics, the
third-integer resonance, sextupole controls and a time-dependent spill.
These lessons adapt the
[Xsuite slow-extraction tutorial](https://github.com/xsuite/slow_extraction_tutorial),
using a shared [native model](../../../../test_data/example_lattices/pimms_extraction/README.md).
Each script loads that model independently.

| Example | Lesson |
| --- | --- |
| [001_resonance_and_septum_optics.py](001_resonance_and_septum_optics.py) | Inspect the near-third-integer operating point and track the three resonance branches at the septum observation plane |
| [002_sextupole_area_and_orientation.py](002_sextupole_area_and_orientation.py) | Change both extraction sextupoles; compare a sampled bounded contour, its area and the outgoing branch slope |
| [003_programmed_spill.py](003_programmed_spill.py) | Ramp the extraction sextupoles, track a coasting distribution and attribute crossings of an ideal septum boundary |

## Boundary and spill interpretation

`001` and `002` do not install an intercepting septum. Their dashed line marks
the proposed x=35 mm boundary at s=0. In `002`, bisection locates the transition
between positive-x, px=0 launches that do or do not reach that boundary within
the chosen tracking horizon. The inner trajectory gives a sampled contour area;
the neighbouring outer trajectory gives the outgoing branch slope, using
passages three turns apart. Changing the horizon or launch resolution changes
the estimate. This is not a precision separatrix solver or an infinite-time
stability boundary. Unscaled Courant-Snyder coordinates carry units sqrt(m),
so their phase-space area is in metres.

`003` installs a rectangular absorbing aperture. It reports losses at that
named aperture separately from all other losses and bins the represented proton
population into a boundary-crossing rate. Surviving particles are shown at the
final observation turn; intercepted particles retain coordinates at their
individual loss turns. A finite sample gives a noisy spill, without collective
effects or feedback. No septum wire, scattering, electrostatic deflection or
extracted trajectory is modelled, so the crossing fraction is not an extraction
efficiency.

The proton operating point and compact geometry differ from the
[mixed carbon/proton acceleration study](../README.md). Both are illustrative
PIMMS configurations, not operational settings.
