<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Custom elements

Write a tracking kernel, add particle-local randomness, record events, or supply
a Python interaction process. These are advanced authoring examples; use existing
Xtrack elements for ordinary lattice modelling. Each script runs independently
on CPU. `001`–`003` require a working C compiler and enable kernel compilation on
their own context, since custom classes are absent from the prebuilt kernels.
The Python callback in `004` does not require writing or compiling a C kernel.

| Example | Lesson |
| --- | --- |
| [001_custom_element.py](001_custom_element.py) | Declare a parameter and a thin quadrupole kernel; compare its kicks with `Multipole` |
| [002_random_kicks.py](002_random_kicks.py) | Include a Gaussian generator, seed particle streams, measure diffusion and replay a copied RNG state |
| [003_internal_records.py](003_internal_records.py) | Declare an event schema, share a record between elements and stop logging independently of tracking |
| [004_python_interaction_process.py](004_python_interaction_process.py) | Supply a CPU callback that returns daughter particles through `BeamInteraction` |

`_xofields`, `_extra_c_sources`, `_depends_on` and `_internal_record_class` are
the element authoring hooks. The C entry point follows the class name, and
`START_PER_PARTICLE_BLOCK` exposes each particle through the `LocalParticle` API.
The examples keep the reference momentum fixed and use normalized canonical
transverse momenta. The quadrupole comparison uses reference-species protons
(`mass_ratio = charge_ratio = 1`). The kernels do not implement backtracking or spin transport.

Explicit RNG seeding currently requires the underscored
`Particles._init_random_number_generator(seeds=...)` initializer. Seeds initialize
the streams in particle storage order; particle IDs do not themselves choose
seeds. Replay here means the same CPU implementation and copied RNG states,
without a claim of bitwise agreement across execution contexts.

Records reserve slots through `RecordIndex_get_slot`. Once capacity is exhausted,
new events are omitted while tracking continues. The default capacity holds all
events; reduce it to see truncation. Read only `record._index.num_recorded` slots.
The record carries particle IDs and element/turn indices, rather than relying on
event ordering. Stopping logging leaves the recorded data available.

Further record layouts and per-element attachment remain in
[element_internal_record](../../element_internal_record/); additional generator
studies remain in [random_number_generator](../../random_number_generator/).
For using existing maps, see [maps](../maps/README.md).

`004` teaches the Python process and product-payload contract, with a prescribed
same-species replacement rather than physical reaction modelling. Xtrack inserts
the returned particles and assigns their identifiers. The lesson uses unit
weights. For storage capacity and `add_particles` independently of tracking, see
[particle insertion](../../particles/011_capacity_and_insertion.py).
