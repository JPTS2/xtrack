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
| [001_custom_element.py](001_custom_element.py) | Compare a compiled thin quadrupole with `Multipole` and a Python tracking object |
| [002_random_kicks.py](002_random_kicks.py) | Choose Gaussian or uniform kicks, initialize native streams and replay copied random state |
| [003_internal_records.py](003_internal_records.py) | Record before/after passage tables with separate capacities; share records in a line or attach them to standalone elements |
| [004_python_interaction_process.py](004_python_interaction_process.py) | Supply a CPU callback that returns daughter particles through `BeamInteraction` |

`_xofields`, `_extra_c_sources`, `_depends_on` and `_internal_record_class` are
the element authoring hooks. The C entry point follows the class name, and
`START_PER_PARTICLE_BLOCK` exposes each particle through the `LocalParticle` API.
The examples keep the reference momentum fixed and use normalized canonical
transverse momenta. The quadrupole comparison uses reference-species protons
(`mass_ratio = charge_ratio = 1`). The kernels do not implement backtracking or spin transport.

`001` also appends a Python object with a `track(particles)` method. The line
schedules this CPU callback separately from compiled tracking sections. It
handles live reference-species particles and demonstrates the same thin kick;
it does not provide compiled-element serialization or GPU execution.

`002` initializes streams through public `RandomNormal.generate` sampling.
The custom class declares `needs_rng = True` so tracking also initializes missing
streams automatically. Gaussian and bounded-uniform kicks have the same chosen
single-passage RMS. Samples from `RandomUniformAccurate` and `RandomExponential`
show their different mean and width conventions. Explicit seed selection is not
part of this lesson. Replay means the same CPU implementation and copied RNG
states, without a claim of bitwise agreement across execution contexts.

`003` declares two passage tables, before and after each kick, with independent
`RecordIndex` fields and capacities. The smaller after table fills first while
kicks continue normally. Read the populated counts from the explicitly declared
`_index` fields; these belong to the author-defined record schema, and no public
count accessor is supplied by Xtrack. Stored particle, element and turn IDs
identify events without relying on execution order.

The line-level logging session shares one record between both kick elements.
The standalone example uses `new_io_buffer` and `start_internal_logging` to
attach another record to selected elements. It stops only one logger, then both;
tracking continues and the recorded data remains available. These attachment
operations do not alter the kicks.

The original [record studies](../../element_internal_record/) retain additional
validation and documentation consumers. Additional generator studies remain in
[random_number_generator](../../random_number_generator/).
For using existing maps, see [maps](../maps/README.md).

`004` teaches the Python process and product-payload contract, with a prescribed
same-species replacement rather than physical reaction modelling. Xtrack inserts
the returned particles and assigns their identifiers. The lesson uses unit
weights. For storage capacity and `add_particles` independently of tracking, see
[particle insertion](../../particles/011_capacity_and_insertion.py).
