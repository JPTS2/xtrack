# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import h5py
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model and weighted launch
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
X               = np.array([-0.1, 0.0, 0.2]) * 1e-3  # [m]
PX              = np.array([1.0, 2.0, -1.0]) * 1e-6  # Normalized momentum
WEIGHTS         = np.array([1.0, 2.0, 1.0]) * 1e6    # Real electrons
FILLING         = np.array([1, 1, 1])

########################################
# Recording duration and script-local output
########################################
N_TURNS         = 12
SAVE_EVERY      = 4
OUTPUT_FILE     = 'outputs/progressive_statistics.h5'

################################################################################
# Prepare one weighted population in the magnetic ring
################################################################################

########################################
# Load a fresh demonstration model
########################################
# Lattice path is relative to the repository root.
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']

########################################
# Place one weighted particle at each prescribed slot center
########################################
# The 1.5 km ring has a 1 GeV/c electron reference and RF voltage zero.
# Radiation is off. This deterministic launch is not an RF-matched bunch.
spacing     = line.get_length() / len(FILLING)
particles   = line.build_particles(
    x       = X,
    px      = PX,
    zeta    = -spacing * np.arange(len(FILLING)),
    weight  = WEIGHTS,
    delta   = 0.0)

# Slot j is centered at zeta=-j*spacing. The 500 m spacing defines recording
# slots, not RF buckets. One particle per slot has zero within-slot rms size.

################################################################################
# Configure a full-length in-memory frame and its HDF5 file
################################################################################

########################################
# Prepare the destination beside this script
########################################
output_path = Path(__file__).resolve().parent / OUTPUT_FILE
output_path.parent.mkdir(parents = True, exist_ok = True)

########################################
# Install the monitor at the first element entrance
########################################
# Passing output_file initializes a new file, replacing this run's old output.
# num_particles is represented intensity: sum(weight), not macroparticle count.
monitor = xt.BeamStatsMonitor(
    start_at_turn       = 0,
    stop_at_turn        = N_TURNS,
    filling_scheme      = FILLING,
    bunch_spacing_zeta  = spacing,
    stats               = ['num_particles', 'mean_x', 'sigma_x'],
    output_file         = output_path)
line.insert('statistics', monitor, at = 'arc.s1.c1.qf.in@start')

################################################################################
# Append newly filled records while tracking continues
################################################################################

########################################
# Save after each tracking interval
########################################
for start in range(0, N_TURNS, SAVE_EVERY):
    interval = min(SAVE_EVERY, N_TURNS - start)
    line.track(particles, num_turns = interval)
    monitor.save_to_file()
    with h5py.File(output_path, 'r') as saved:
        print('Completed turns / saved rows:', particles.at_turn[0],
              len(saved['turns']))

########################################
# Save again without recording any new turns
########################################
monitor.save_to_file()
with h5py.File(output_path, 'r') as saved:
    saved_turns = saved['turns'][...]
    saved_mean  = saved['stats/beam/mean_x'][...]
    saved_sigma = saved['stats/beam/sigma_x'][...]
    saved_count = saved['stats/beam/num_particles'][...]
    bunch_mean  = saved['stats/bunch/mean_x'][...]
    bunch_count = saved['stats/bunch/num_particles'][...]
print('Rows after the repeated save:', len(saved_turns))

################################################################################
# Compare the saved history with the full in-memory frame
################################################################################

########################################
# Inspect turn labels and weighted moments
########################################
beam_mean   = monitor.get('mean_x', level = 'beam')
xt.Table(
    data    = {
        'turn':          saved_turns,
        'mean_memory_m': beam_mean,
        'mean_file_m':   saved_mean,
        'sigma_file_m':  saved_sigma,
        'intensity':     saved_count},
    index   = 'turn').show(maxwidth = 'full')

########################################
# Compare all saved rows numerically
########################################
print('Maximum turn-label difference:',
      np.max(np.abs(saved_turns - monitor.turns)))
with h5py.File(output_path, 'r') as saved:
    for level in monitor.available_levels:
        for statistic in monitor.stats:
            memory      = monitor.get(statistic, level = level)
            from_file   = saved[f'stats/{level}/{statistic}'][...]
            print(level, statistic, 'maximum file/memory difference:',
                  np.max(np.abs(from_file - memory)))
print('Saved beam/bunch mean shapes:', saved_mean.shape, bunch_mean.shape)
print('Selected slots:', monitor.selected_slots)
print('Final bunch means [m] and represented populations:',
      bunch_mean[-1], bunch_count[-1])
print('Saved statistics:', output_path)

# Saving writes only the newly filled suffix. It neither clears nor shrinks
# the configured frame. Beam reductions pool the accepted slots' raw moments;
# the beam rms includes their distinct centroids, unlike the zero bunch widths.
