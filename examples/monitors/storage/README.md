<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Saving monitor statistics

Save statistics during tracking, then reuse a bounded in-memory frame. Each
script runs independently and requires optional **h5py** for HDF5 output.

| Example | Lesson |
| --- | --- |
| [001_save_statistics_during_tracking.py](001_save_statistics_during_tracking.py) | Append newly filled records while retaining the complete configured frame in memory |
| [002_reuse_recording_frames.py](002_reuse_recording_frames.py) | Save, clear and retarget one fixed-size frame while accumulating a disk history |

Both use a fresh [demonstration ring](../../../test_data/example_lattices/demo_ring/README.md)
with its 1 GeV/c electron reference. RF voltage is zero and radiation is off.
Three deterministic weighted particles keep the storage comparison visible;
they are not a statistical or RF-matched bunch. `001` puts one particle at each
of three prescribed slot centers, 500 m apart in zeta. These are recording
slots, not RF buckets; the one-particle bunch widths are zero. The monitor samples the ring
entrance before each turn. `num_particles` is the sum of weights.

`001` records twelve turns and saves after intervals of four. A repeated save
without new tracking appends no duplicate rows. All configured records remain
in memory: saving is not a memory-release operation. Beam datasets have shape
`(turn,)` and bunch datasets `(turn, selected slot)`. Both are compared with
in-memory statistics; the pooled beam width includes the distinct slot centroids.

`002` records four turns at a time. It saves each interval before calling
`start_new_frame`, which clears the in-memory data and moves the same-size
recording window to later turns. It compares each saved interval numerically
before the next reset; the final file contains one flat twelve-turn history.
This second example uses whole-population monitoring. Neither example uses
the monitor's coasting mode.

Each run initializes its own file under `outputs/` beside the script, replacing
its previous output. Datasets under `stats/beam/` and, for `001`, `stats/bunch/` hold the requested
statistics; `turns` holds absolute turn labels. The examples compare those datasets with
monitor values, including weighted intensity and rms size. They do not store
individual particle trajectories.

For weighted moments and emittance interpretation, return to
[derived statistics](../beam_statistics/002_derived_statistics.py).
