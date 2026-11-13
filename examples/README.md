<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Xtrack examples

For installation, the API reference and the physics guide, see the
[Xsuite documentation](https://xsuite.readthedocs.io/en/latest/index.html).
Start with [getting started](getting_started/README.md) for a runnable tour;
each step points to a subject where you can learn more.

The collection groups operations into subjects, with subfolders for distinct
physical problems. Integrated RF, radiation, spin and machine workflows keep
their connected calculations together. Each subject guide gives a reading route,
the physical assumptions and any additional dependencies.

For detailed companion-package physics examples, see
[Xfields](https://github.com/xsuite/xfields/tree/main/examples) for space charge
and beam–beam interactions,
[Xcoll](https://github.com/xsuite/xcoll/tree/main/examples) for material
interactions and collimation, and
[Xwakes](https://github.com/xsuite/xwakes/tree/main/examples) for wakes and
impedance-driven beam dynamics.

## Folder layout

```text
examples/
├── getting_started/
├── tables/
├── line_construction/
├── elements/
│   ├── magnets/
│   ├── cavities/
│   ├── maps/
│   ├── transformations/
│   └── custom/
├── geometry_and_survey/
├── particles/
├── optics/
│   ├── transverse/
│   ├── longitudinal/
│   ├── nonlinear/
│   └── collective/
├── matching/
│   └── longitudinal/
├── tracking/
│   ├── time_dependent/
│   ├── nonlinear/
│   └── collective/
├── monitors/
│   ├── particle_records/
│   ├── beam_statistics/
│   └── storage/
├── rf/
│   ├── dynamics/
│   ├── timing/
│   └── acceleration/
├── radiation/
├── spin/
├── errors_and_correction/
├── apertures/
├── contexts_and_memory/
├── interfaces/
│   ├── madng/
│   └── rf_track/
├── machine_studies/
│   ├── fcc_ee/
│   ├── lhc/
│   ├── elena/
│   ├── pimms/
│   ├── psb/
│   └── lep/
└── benchmarks/
    ├── performance/
    ├── convergence/
    └── statistics/
```

## getting_started

The [introductory tour](getting_started/README.md) flows through line
construction, inspection, open and periodic optics, matching, tracking,
measurements and persistence. The subject folders also have their own starting
points; the tour is not a prerequisite for every application.

## tables

[Tables](tables/README.md) covers named rows, column/cell access, selection,
calculation and scalar metadata. Persistence includes native dictionaries,
JSON/CSV, TFS layouts and separate HDF5 storage. TFS is a data format here;
no MAD-X executable is required.

## line_construction

[Line construction](line_construction/README.md) develops environments, ordered
assembly, anchored placement, strength/geometry expressions and sequence editing.
It distinguishes shared elements, cloning, replicas and imported models, then
covers slicing, named reference particles and persistence. Native
[model definitions](../test_data/example_lattices/README.md) support the lessons.
Legacy MAD-X-format loading is a data-input route; new models use native Python.

## elements

[Elements](elements/README.md) explains individual field and transport models:
[magnets](elements/magnets/README.md),
[cavities](elements/cavities/README.md),
[maps](elements/maps/README.md) and
[spatial transformations](elements/transformations/README.md).
The [electron cooler](elements/electron_cooler.py) demonstrates one native
friction kick. [Custom elements](elements/custom/README.md) covers tracking
kernels, random streams, event records and Python interaction processes.
Machine-level RF dynamics and apertures have their own subjects.

## geometry_and_survey

[Geometry and survey](geometry_and_survey/README.md) connects the reference
path, global placement, actual element frames and tracked coordinates. It
includes nonplanar frame changes and rectangular-bend face/slicing conventions.
Reference geometry and the physical closed orbit are distinct quantities.

## particles

[Particles](particles/README.md) starts with reference quantities and coordinates,
then distributions and matched preparation. Container operations cover copying,
serialization, selection, IDs, merging and capacity; mixed-species launches
extend the reference conventions to ions and different charge-to-mass ratios.

## optics

- [Transverse](optics/transverse/README.md): periodic/open Twiss, closed orbit,
  initial conditions, dispersion, beam sizes, matrices and coupled modes.
- [Longitudinal](optics/longitudinal/README.md): 6D optics, synchrotron motion,
  longitudinal derivatives and crab-cavity coupling.
- [Nonlinear](optics/nonlinear/README.md): chromatic optics, detuning and
  resonance-driving terms.
- [Collective-element boundaries](optics/collective/README.md): prescribed
  frozen fields and the treatment of updating collective elements in Twiss.

## matching

[Matching](matching/README.md) has routes through physical targets and optimizer
tools: tunes, transport, bumps, optics errors, chromaticity, coupling and shared
magnets; constraints, custom Actions, histories and generated knobs.
[Longitudinal matching](matching/longitudinal/README.md) connects RF controls to
synchrotron tune and synchronous-state targets.

## tracking

[Tracking](tracking/README.md) covers propagation, partial paths, backtracking,
coordinate freezing and element-by-element records.
[Time-dependent tracking](tracking/time_dependent/README.md) separates clock
functions, programmed bumps and numeric family updates.
[Nonlinear tracking](tracking/nonlinear/README.md) covers footprints, islands
and dynamic aperture. [Collective boundaries](tracking/collective/README.md)
teaches Xtrack scheduling and record contracts; full solver workflows belong
to their implementing packages.

## monitors

[Monitors](monitors/README.md) separates
[particle records](monitors/particle_records/README.md),
[beam statistics](monitors/beam_statistics/README.md) and
[storage](monitors/storage/README.md). It covers observation locations,
acquisition windows, profiles and derived statistics, finite buffers and
retrieval. Tracking and monitor configuration have separate reading routes.

## rf

[RF](rf/README.md) connects three subjects:
[dynamics](rf/dynamics/README.md) for buckets, transition and slipping RF systems;
[timing](rf/timing/README.md) for absolute-time cavities, delay and pickups;
[acceleration](rf/acceleration/README.md) for reference programmes, adiabatic
motion and linac transport. Single-cavity behaviour starts with
[elements](elements/cavities/README.md).

## radiation

[Radiation](radiation/README.md) connects energy loss and RF compensation,
tapering, integrals, damping, equilibrium and emitted photons. Separate
applications inspect radiation linearizations, wigglers/combined fields and
energy extraction. Equilibrium predictions, finite tracking and statistical
convergence answer different questions.

## spin

[Spin](spin/README.md) progresses from precession and open transport to periodic
spin fields, ring tracking and spin-tune dependence. Magnetic excitation and
resonance sweeps form one route; radiative equilibrium and decoherence form
another. An invariant axis, a physical polarization vector and an ensemble mean
are distinct observables.

## errors_and_correction

[Errors and correction](errors_and_correction/README.md) covers alignment and
multipole errors, orbit/trajectory responses, BPM effects, threading, corrector
limits and optics correction. It distinguishes correcting a measured orbit
from recovering dispersion, optics or beam performance.

## apertures

[Apertures](apertures/README.md) starts with tracking cuts, chamber profiles,
SVG shapes, pipe placement and curved/overlapping geometry. It then connects
installed or associated aperture data, tracking limits, losses/refinement and
beam-envelope clearance. A geometric chamber, a numerical tracking guard and
physical material interception have different meanings.

## contexts_and_memory

[Contexts and memory](contexts_and_memory/README.md) covers execution contexts,
CPU/device transfers, array snapshots, buffers and shared versus independent
storage. Optional GPU configurations need their packages, devices and drivers;
small transfer examples do not measure throughput.

## interfaces

[Interfaces](interfaces/README.md) connects external codes:
[MAD-NG](interfaces/madng/README.md) for optics, matching and normal forms;
[RF-Track](interfaces/rf_track/README.md) for external elements and field-map
transport. Each route states the verified reference/coordinate contracts and
its optional engine dependencies.

## machine_studies

[Machine studies](machine_studies/README.md) carries models through connected
calculations: FCC-ee IP optics/errors/emittance and detector fields; LHC shared
magnets and collision optics; ELENA curved fields; PIMMS mixed-species
acceleration and extraction; PSB injection; LEP spin and polarization.
The shared demonstration ring provides a compact integrated starting workflow.
Each machine guide states its particular model and operating point.

## benchmarks

[Benchmarks](benchmarks/README.md) groups
[performance](benchmarks/performance/README.md),
[convergence](benchmarks/convergence/README.md) and
[statistics](benchmarks/statistics/README.md). These studies make timing
boundaries, numerical references and statistical uncertainty explicit.
Their workloads are separate from ordinary teaching-example execution.
