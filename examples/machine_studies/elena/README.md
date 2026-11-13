<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# ELENA: curved magnetic fields

Use the same fitted-field ring to connect reference geometry, magnetic errors
and tracking resolution. The reference is a 100 MeV/c antiproton; all three
examples use the same geometry and powered quadrupoles.

| Example | Lesson |
| --- | --- |
| [001_field_transport.py](001_field_transport.py) | Inspect fitted dipole fields, reference curvature, periodic optics and the resulting closed orbit |
| [002_dipole_field_error.py](002_dipole_field_error.py) | Scale one complete fitted dipole field and compare orbit and beta beating while the reference geometry stays fixed |
| [003_tracking_resolution.py](003_tracking_resolution.py) | Track identical physical launches at different RK4 resolutions and compare normalized endpoints |

The [native model](../../../test_data/example_lattices/elena_field/README.md)
retains six 60-degree bends, fitted fringe fields extending into straight
reference sections and twelve powered quadrupoles. The field plots show
rigidity-normalized fields, not tesla. The fit's source map is not included;
these examples use its stored coefficients without refitting them.

RF, apertures and cooling are absent. The field error is an illustrative current
scaling, not a measured ELENA error. The tracking comparison isolates numerical
integration of that fitted model; it does not measure field-fit accuracy or
establish long-term stability. RK4 is not exactly symplectic.

For the individual element, see [field expansions](../../elements/magnets/README.md).
