<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Collective tracking

Track a bunch through an updating source, then coordinate two interacting
bunches. Both scripts require **Xfields** in addition to Xtrack and run
independently on a single CPU process.

| Example | Lesson |
| --- | --- |
| [001_track_collective_elements.py](001_track_collective_elements.py) | Compare prescribed and bunch-updated Gaussian space-charge fields |
| [002_pipeline_scheduling.py](002_pipeline_scheduling.py) | Exchange actual bunch moments through pipeline holds and resumes |

`001` uses the [shared demonstration ring](../../../test_data/example_lattices/demo_ring/README.md)
with its 1 GeV/c electron reference and 1.5 km circumference. Identical weighted
bunch samples pass through
independent frozen and updating `SpaceChargeBiGaussian` elements. Both fields
act; the updating source fits the full bunch's transverse moments at each pass.
The example checks the applied moments and compares final trajectories by ID.
Its printed central-force estimate makes the relativistic force scale explicit.

The thin interaction has an effective length of 1 m without adding geometry.
It illustrates Gaussian moment closure at one location, with a prescribed
longitudinal profile and no longitudinal space-charge kick. The sampled
covariance follows the external optics; it is not a space-charge equilibrium.
RF voltage is zero and radiation is off. This is not a distributed-ring or PIC
model. Tracking separate chunks through the updating element would fit different
sources: the full bunch is the collective input.

`002` uses two separate 7 TeV/c proton populations. Each branch contains one
head-on `BeamBeamBiGaussian2D` interaction and a linear one-turn map. These are
explicit analytical maps, rather than a geometric collider lattice. The
single-process pipeline exchanges partner moments before completing each
collision. Actual scheduler holds, completed turns and received pre-collision
centroids are displayed. No MPI installation is needed.

The qualified case uses uncoupled, reflection-symmetric transverse samples.
Opposing local frames reverse the partner's horizontal centroid. General tilted
Gaussian profiles and their covariance transformations are outside this lesson.
Scheduling is provided by Xtrack; the Gaussian interaction and moment update
are Xfields physics. No MPI scaling or Gaussian-model accuracy is established.

These two-population interactions differ from tracking several particle species
inside one container. For linearization and collective-element selection in
Twiss, see [collective optics](../../optics/collective/).

For distributed frozen, quasi-frozen and PIC space charge, follow the
[Xsuite collective-effects guide](https://xsuite.readthedocs.io/en/latest/collective.html#space-charge)
and the [Xfields space-charge examples](https://github.com/xsuite/xfields/tree/main/examples/001_spacecharge).
These cover different physical models from the single thin Gaussian source in
`001`. For wakes, dampers and collective-instability studies, use the
[Xwakes examples](https://github.com/xsuite/xwakes/tree/main/examples).
The companion packages own those physics workflows; this folder teaches how
Xtrack schedules their collective interactions.
