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
| [004_freeze_longitudinal.py](004_freeze_longitudinal.py) | Temporarily freeze longitudinal coordinates, then resume their evolution |
| [005_optimize_for_tracking.py](005_optimize_for_tracking.py) | Use optimize_for_tracking on an independent copy and inspect the resulting changes |

`001` inspects the final particle state, rather than recording its history.
Complete turns reset `s` to zero. `002` deliberately keeps `s` cumulative so
the traversed distance remains visible across line-end crossings. Its turn
counter still counts those crossings, not the number of tracking calls.

`003` applies inverse element maps to the same forward intervals. This does
not model a beam travelling in the opposite direction through the fields.

`004` enables compilation only on its own CPU context because the frozen
configuration may need a new kernel. It requires a working C compiler.
Freezing is a numerical constraint, not longitudinal confinement. In this
static magnetic ring delta is conserved even without freezing; zeta evolves
again after the frozen call. Equal transverse endpoints are specific to this
model, whose transverse maps depend on delta but not on zeta.
The demonstrated momentum offsets are within `+/-1e-4`; this is a tested
launch sample, not a momentum-acceptance measurement. The bare demonstration
ring has large chromaticity and a vertical tune near an integer.

`005` compares a fixed configuration on an independent deep copy. Optimization
disables expressions for the copy's entire environment and can remove markers
and change indices. Passive BPM Devices become drifts and lose their survey
poses; their zero-length observation planes disappear. Inactive multipoles
are also removed, so their circuits cannot subsequently be powered. The
original environment remains live; retained numeric fields in the snapshot
remain editable. Coordinate agreement is checked for this nominal aligned
model and launch, without a tracking-throughput claim.

Except for the transparent proton drift line in `002`, the examples use the
[shared demonstration ring](../../test_data/example_lattices/demo_ring/README.md)
with its 1 GeV/c electron reference and 1.5 km circumference. RF voltage and
nonlinear strengths are zero; radiation and collective forces are absent.
Backtracking inverts deterministic maps; it does not undo losses or stochastic
processes.

For trajectories, beam sizes and recorded observations, continue with
[monitors](../monitors/). For geometrical cuts and loss positions, see
[the apertures plan](../README.md#apertures).

The [examples index](../README.md#tracking) also lays out time-dependent,
nonlinear, collective and beam-interaction tracking. CPU/GPU contexts and
memory ownership belong to [contexts and memory](../README.md#contexts_and_memory).

For clock-driven machine settings and sampled transverse drives, continue with
[time-dependent tracking](time_dependent/README.md).
