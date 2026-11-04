<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Monitors

Choose an observation according to the question: motion over successive turns,
a beam envelope along the lattice, or measurements at selected locations.
The examples run independently using the
[native magnetic ring](../../test_data/example_lattices/README.md).
For propagation controls, see [tracking](../tracking/).

| Example | Question |
| --- | --- |
| [001_turn_by_turn.py](001_turn_by_turn.py) | How do saved samples differ from the current particle coordinates? |
| [002_element_by_element.py](002_element_by_element.py) | Does a tracked bunch follow the beam envelope predicted by Twiss along the lattice? |
| [003_observation_points.py](003_observation_points.py) | How do installed particle monitors and named-location recording compare? |
| [004_valid_samples.py](004_valid_samples.py) | How do I interpret missing samples and particle reordering after a loss? |
| [005_recording_windows.py](005_recording_windows.py) | How do I select particle IDs and turns, then sample repeated windows? |
| [006_derived_statistics.py](006_derived_statistics.py) | How do weighted moments determine beam size, projected emittance and normal-mode emittances? |

For a first turn-by-turn plot, see
[getting started](../getting_started/06_track_particles.py).
`001` develops that introduction by separating a saved observation interval
from subsequent unmonitored propagation.
`001`–`004` cover particle records and validity. `005` controls which particles
and turns are recorded. `006` then moves from individual records to weighted
population measurements: moments, beam sizes and covariance-derived emittances.

`002` uses `turn_by_turn_monitor = 'ONE_TURN_EBE'`. It records every element
entrance and the final line boundary, then compares measured transverse rms
sizes with `sqrt(beta * geometric_emittance)`. Its beam is on momentum and the
model is uncoupled; finite sampling and nonlinear maps limit agreement with
linear Twiss. This is a one-turn transverse envelope comparison, not a matched
longitudinal bunch.

The particle records in `001`–`005` store individual coordinates.
Their sample axes differ: `ParticlesMonitor` uses `(particle-ID slot, sample)`,
whereas a multi-element record's `get('x')` uses
`(turn, particle-ID slot, observation)`.
Repeated particle-monitor windows add a leading frame axis. Window starts are
inclusive and stops are exclusive. `005` first teaches one window, then shows
explicit sampling in a manual turn loop and repeated windows. Monitor dictionary
round trips preserve configuration, not recorded trajectories.

`006` uses native `BeamStatsMonitor` to accumulate weighted moments rather than
individual trajectories. It uses one prepared distribution throughout: first
calculate its weighted central moments and projected emittance directly, then
compare with the monitor and rotate the initial distribution canonically.
`num_particles` means sum of weights.
Projected emittances have the `_projected` suffix; unqualified emittance names
refer to covariance normal modes. `006` adds independent longitudinal coordinates
to a transverse launch for full-rank instantaneous covariance measurements.
This synthetic distribution is not RF equilibrium or a dispersively matched
beam. It uses a canonical transverse rotation to expose the distinction between
projections and eigenemittances.

`004` uses a simple aperture cut only to produce an incomplete record. Its lesson
is identifying particles by ID and selecting valid recorded states. It does not
teach aperture design or loss refinement.

These ring examples have a 1 GeV/c electron reference and no RF, radiation or
collective forces. The model has no explicit aperture except the cut in `004`.
For the existing bunch/slice/coasting, profile and HDF5 examples, see
[beam-statistics monitors](../beam_stats_monitor/).
