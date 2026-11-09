<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Particle records

Record individual trajectories over turns, along a line or at selected
locations. Each example runs independently on the shared
[demo ring](../../../test_data/example_lattices/demo_ring/README.md), with a
1 GeV/c electron reference, zero RF voltage and no radiation.

| Example | Lesson |
| --- | --- |
| [001_turn_by_turn.py](001_turn_by_turn.py) | Separate saved entrance samples from a later particle state |
| [002_element_by_element.py](002_element_by_element.py) | Compare a tracked one-turn beam envelope with linear Twiss |
| [003_observation_points.py](003_observation_points.py) | Compare named-location records with installed monitors at IP BPM planes |
| [004_valid_samples.py](004_valid_samples.py) | Follow particle IDs and distinguish valid samples from unfilled slots after a loss |
| [005_recording_windows.py](005_recording_windows.py) | Select ID and turn ranges, sample manually and repeat recording windows |

`ParticlesMonitor` arrays use `(particle-ID slot, sample)`. Named-location
records use `(turn, particle-ID slot, observation)` when read with `get('x')`.
Repeated windows add a leading frame axis. Starts are inclusive and stops
exclusive; a dictionary round trip preserves configuration, not recorded data.

`002` records every element entrance and the final line boundary. Its finite
on-momentum sample follows the transverse linear envelope approximately; it
is not a matched longitudinal bunch. `004` adds a single aperture only to
produce an incomplete record, without an aperture-design or loss-refinement
claim. Use particle IDs and recorded states when associating samples with
launches after tracking.

For weighted population moments and covariance-derived emittances, continue
with [beam statistics](../beam_statistics/). See [monitors](../README.md) for
the wider observation route and [tracking](../../tracking/) for propagation.
