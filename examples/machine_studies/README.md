<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Machine studies

Carry a setting through several operations while keeping its model and physical
assumptions explicit. These workflows connect the independent subject lessons.

| Example | Lesson |
| --- | --- |
| [demo_ring_workflow.py](demo_ring_workflow.py) | Correct global chromaticities, inspect off-momentum IP optics and track common launches at the same operating point |

The workflow uses the shared 1 GeV/c electron demonstration ring. It retains an
independent bare control and varies only the installed sextupole circuits.
The corrected environment is then used directly for both off-momentum optics
and tracking, without loading a fresh setting between operations.

Global tune-derivative targets leave higher-order chromaticity and insertion
chromatic beta unconstrained. The example makes those limitations visible;
its nine launches over a short horizon do not establish a momentum acceptance
or dynamic aperture. Both settings receive identical physical entrance
coordinates prepared from the bare optics.

For the individual operations, see [chromaticity matching](../matching/010_match_chromaticity.py),
[chromatic optics](../optics/nonlinear/001_chromatic_optics.py) and
[nonlinear tracking](../tracking/nonlinear/README.md). The
[model definition](../../test_data/example_lattices/demo_ring/README.md) gives
the geometry, nominal strengths and qualification boundaries.
