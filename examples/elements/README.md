<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Elements

Learn how individual elements represent fields, geometry and particle transport.
For assembling these elements into a model, use
[line construction](../line_construction/README.md).

| Topic | Lessons |
| --- | --- |
| [Magnets](magnets/README.md) | Fields and strengths, models/integrators, slicing, bends, fringes, solenoids and field-based tracking |
| [Cavities](cavities/README.md) | Energy and transverse RF kicks, finite-length transport and numerical map choices |
| [Maps](maps/README.md) | Taylor approximations, assigned segment optics, prescribed noise and RF-array controls |
| [Spatial transformations](transformations/README.md) | Coordinate frames, ordered rotations and element misalignment attributes |
| [Electron cooler](electron_cooler.py) | Apply one native friction kick and compare powered and zero-current particles |
| [Custom elements](custom/README.md) | Tracking kernels, particle random streams, event records and Python interaction processes |

Machine-level RF dynamics belongs with RF; apertures have their own subject.

The electron-cooler lesson uses a homogeneous positive-ion reference. It compares
one passage with zero and nonzero electron current; the configured length
integrates the friction force without adding drift transport. The guiding
magnetic field enters that friction model rather than a solenoid optics map.
This single-element comparison does not determine a cooling time or equilibrium.
