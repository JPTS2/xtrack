<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Machine studies

Carry a machine model through connected calculations: choose an operating
point, change its settings and inspect the consequences for optics, tracking
or equilibrium. The subject folders introduce the individual APIs; these routes
show how they fit together when the machine geometry and controls matter.

| Route | Species | Connected studies |
| --- | --- | --- |
| [Demonstration ring](demo_ring_workflow.py) | Electrons | Chromaticity correction, off-momentum IP optics and common-launch tracking |
| [FCC-ee](fcc_ee/README.md) | Positrons | IP beta control, alignment sensitivity, orbit correction and equilibrium emittance; separate detector-solenoid insertion |
| [LHC](lhc/README.md) | Protons | Counter-rotating geometry, shared magnets, collision knobs and joint two-beam optics matching |
| [ELENA](elena/README.md) | Antiprotons | Curved field expansion, field-error orbit/optics response and tracking-resolution comparison |
| [PIMMS](pimms/README.md) | Carbon ions and protons | Common-rigidity acceleration and pickups; separate resonant-extraction optics, sextupole controls and spill |
| [PSB](psb/README.md) | Protons | Injection chicane, time-dependent optics correction and multiturn injection tracking |
| [LEP](lep/README.md) | Positrons | Detector-induced spin/orbital response, spin bumps, coupling correction and polarization recovery |

Each guide states the model and physical scope. These stored design and teaching
models are not current operational settings. Different operating points remain
explicit: FCC-ee's Z-ring and FCC-t detector insertion use distinct models;
PIMMS acceleration and extraction use distinct reference/geometry configurations.
Model-import and independent reference comparisons are separate from these
reading routes.

## Demonstration-ring workflow

The [workflow](demo_ring_workflow.py) uses the shared 1 GeV/c electron ring.
It retains a bare control, corrects global chromaticities through the sextupole
circuits, then carries that same setting into off-momentum optics and tracking.
Both cases receive identical physical entrance coordinates from the bare optics.

Tune-derivative targets leave higher-order chromaticity and insertion chromatic
beta unconstrained. Nine launches over a short horizon do not establish momentum
acceptance or dynamic aperture. For the individual operations, see
[chromaticity matching](../matching/010_match_chromaticity.py),
[chromatic optics](../optics/nonlinear/001_chromatic_optics.py) and
[nonlinear tracking](../tracking/nonlinear/README.md). The
[model definition](../../test_data/example_lattices/demo_ring/README.md)
gives the geometry, controls and qualified configurations.
