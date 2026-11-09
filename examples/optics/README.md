<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Optics

Calculate and interpret optics, starting with periodic and open Twiss solutions.
The collection separates transverse, longitudinal and nonlinear optics; coupled
mode interpretation belongs here, while constructing coupling bumps and knobs
belongs with matching.

| Route | Available lessons |
| --- | --- |
| [Transverse optics](transverse/README.md) | Twiss, covariance, coupling, boundaries, closed orbits, transfer maps, calculation settings and persistence |
| [Longitudinal optics](longitudinal/README.md) | Synchronous 6D optics, longitudinal covariance and revolution periods |
| [Nonlinear optics](nonlinear/README.md) | Chromaticity, amplitude detuning, multi-turn island optics and stability diagrams |
| [Collective-element handling](collective/README.md) | Prescribed frozen-field optics and default skipping of updating collective forces |

See the [full optics structure](../README.md#optics) for the recipe index.
All chromatic optics, including first-order chromaticity, belongs in the
nonlinear route. Each lesson states its model qualifications; a magnetic 4D ring
is not a universal model for these subjects.

The [getting-started tour](../getting_started/README.md) provides a shorter first
pass. General table operations live in [tables](../tables/README.md).
