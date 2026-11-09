<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Monitors

Record individual trajectories, measure weighted beam statistics, or save
measurements while tracking. Each example runs independently.

| Folder | Lesson |
| --- | --- |
| [particle_records/](particle_records/README.md) | Turn-by-turn and element-by-element trajectories, observation points, valid samples and recording windows |
| [beam_statistics/](beam_statistics/README.md) | Weighted moments, covariance optics, bunches, slices, coasting beams and profiles |
| [storage/](storage/README.md) | Incremental HDF5 saving and bounded recording frames |

For a first turn-by-turn plot, start with
[getting started](../getting_started/06_track_particles.py).
For propagation controls, see [tracking](../tracking/README.md).

Particle records retain individual coordinates. `BeamStatsMonitor` accumulates
weighted population moments and profiles instead; its `num_particles` statistic
is the sum of macroparticle weights. The statistics examples explain which
population and longitudinal interval each measurement represents.

The element-by-element lesson uses `ONE_TURN_EBE` and compares a tracked bunch
with the beam sizes predicted by Twiss. The derived-statistics lesson separates
projected emittances from covariance normal modes. These measurements describe
the supplied distribution; they do not by themselves establish equilibrium.

The storage examples require **h5py**. Model assumptions and any additional
dependencies are stated with the relevant lesson.
