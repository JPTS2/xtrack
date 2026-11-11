<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Contexts and memory

Choose where tracking runs, transfer arrays and particles, then distinguish
storage allocation from shared objects. Each example runs independently and
reports the relevant changes in the terminal.

| Example | Lesson |
| --- | --- |
| [001_cpu_context_and_rebuilds.py](001_cpu_context_and_rebuilds.py) | Select serial or OpenMP CPU execution and explicitly rebuild after editing a line |
| [002_move_particles_between_contexts.py](002_move_particles_between_contexts.py) | Convert arrays, copy or move particles, and resume tracking after a round trip |
| [003_buffers_and_shared_storage.py](003_buffers_and_shared_storage.py) | Allocate objects in a shared buffer and distinguish independent fields from shared line elements |

`001` uses the shared [demonstration ring](../../test_data/example_lattices/demo_ring/README.md).
The default is serial CPU execution; change `N_THREADS` to select OpenMP. The
marker edit adds no length or kick. The rebuilt tracker uses the explicitly
selected context, and the original particle container continues tracking.

`002` defaults to serial and OpenMP CPU contexts. Both use host memory. The
commented `ContextCupy` and `ContextPyopencl` alternatives require the corresponding
optional Python package, a supported device and its drivers. Those GPU
configurations have not been verified here. The tiny samples illustrate
execution and transfer semantics, rather than performance.

Array conversion uses `copy=True` for independent snapshots: on CPU, conversion
without a copy can return the original array. `Particles.copy` allocates another
container; `Particles.move` changes the existing object's storage. After moving,
read arrays through the moved object again. Lost-particle states and IDs travel
with the particles; row order can change, so compare by `particle_id`. The round
trip demonstrates continued tracking of live particles while losses stay lost.

`003` makes independent allocations in one buffer. Editing one object's fields
leaves the other allocations unchanged. A shallow line copy instead shares its
elements, so editing an element through either line affects both. A deep copy
has independent elements. The documented `_context` and `_buffer` constructor
keywords select execution and storage.

For propagation, partial paths and backtracking, see
[tracking](../tracking/README.md). For particle identity and storage capacity,
see [particles](../particles/README.md).
