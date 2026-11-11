<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Machine studies

Carry a setting through several operations while keeping its model and physical
assumptions explicit. Start with the shared demonstration ring, then use the
machine folders for workflows that depend on a particular geometry or operating
point. Each folder explains its input model and what the calculation establishes.

| Example | Lesson |
| --- | --- |
| [demo_ring_workflow.py](demo_ring_workflow.py) | Correct global chromaticities, inspect off-momentum IP optics and track common launches at the same operating point |
| [elena/](elena/README.md) | Follow fitted dipole fields through strongly curved reference sections using `BFieldExpansion` |
| [fcc_ee/](fcc_ee/README.md) | Compensate a tilted detector solenoid in an experimental insertion |
| [lep/](lep/README.md) | Recover predicted equilibrium polarization with installed spin bumps and coupling corrections |
| [lhc/](lhc/README.md) | Apply crossing and separation knobs and inspect the resulting collision optics and local orbit bumps |

These are bounded studies using repository models, rather than current machine
settings. The subject folders teach the individual APIs; these workflows show
how the operations fit together. Other machine-specific lessons may be added
when they contribute a distinct workflow.

## Demonstration-ring workflow

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
