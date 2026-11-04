<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Tracking

These independent examples explain how particles propagate through a line.
Start with complete turns, then select partial paths and reverse them.

| Example | Subject |
| --- | --- |
| [001_track_line.py](001_track_line.py) | Propagate particles and interpret the final element, turn and position |
| [002_partial_tracking.py](002_partial_tracking.py) | Stop, resume, cross the line end and count elements or turns |
| [003_backtrack.py](003_backtrack.py) | Invert full turns and partial paths; inspect coordinate residuals |

[004_execution_context.py](004_execution_context.py) is a separate execution
recipe: select a CPU context, edit the line and retain that context when optics
rebuilds the tracker. It defaults to serial CPU; positive `N_THREADS` selects
OpenMP and requires a compatible compiler/runtime. Xobjects also provides
`ContextCupy` for CUDA and `ContextPyopencl` for OpenCL. These examples do not
qualify those optional backends or measure tracking throughput.

Except for the transparent proton drift line in `002`, the examples use the
[native magnetic ring](../../test_data/example_lattices/README.md) with its
1 GeV/c electron reference. There is no RF, radiation or collective force.
Backtracking inverts deterministic maps; it does not undo losses or stochastic
processes.

For trajectories, beam sizes and recorded observations, continue with
[monitors](../monitor/). For geometrical acceptance and losses, see
[apertures](../apertures/) and [loss refinement](../collimation/).
