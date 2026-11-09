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

########################################
# Fixed frame size and script-local output
########################################
FRAME_TURNS     = 4
N_FRAMES        = 3
OUTPUT_FILE     = 'outputs/reused_frames.h5'

################################################################################
# Prepare one population and a bounded recording frame
################################################################################

########################################
# Load a fresh model and prepare the weighted launch
########################################
# Lattice path is relative to the repository root.
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
particles   = line.build_particles(
    x       = X,
    px      = PX,
    weight  = WEIGHTS,
    delta   = 0.0)

# RF voltage is zero and radiation is off for this 1 GeV/c electron ring.
# The deterministic population is not an RF-matched distribution.

########################################
# Initialize the destination and install a four-turn frame
########################################
output_path = Path(__file__).resolve().parent / OUTPUT_FILE
output_path.parent.mkdir(parents = True, exist_ok = True)

# The constructor creates/replaces this example's output file. The monitor
# stores FRAME_TURNS records; saving does not clear them automatically.
monitor = xt.BeamStatsMonitor(
    start_at_turn   = 0,
    stop_at_turn    = FRAME_TURNS,
    stats           = ['num_particles', 'mean_x', 'sigma_x'],
    output_file     = output_path)
line.insert('statistics', monitor, at = 'arc.s1.c1.qf.in@start')

################################################################################
# Save each frame before clearing it for reuse
################################################################################

########################################
# Retarget the same monitor and append each new interval
########################################
for frame in range(N_FRAMES):
    start = frame * FRAME_TURNS
    if frame > 0:
        monitor.start_new_frame(start_at_turn = start)
    line.track(particles, num_turns = FRAME_TURNS)
    monitor.save_to_file()

    # Read only this interval to compare it before the next frame clears RAM.
    with h5py.File(output_path, 'r') as saved:
        file_turns  = saved['turns'][-FRAME_TURNS:]
        file_mean   = saved['stats/beam/mean_x'][-FRAME_TURNS:]
        file_sigma  = saved['stats/beam/sigma_x'][-FRAME_TURNS:]
        file_count  = saved['stats/beam/num_particles'][-FRAME_TURNS:]
        saved_rows  = len(saved['turns'])
    print('Frame start / saved rows / in-memory records:',
          start, saved_rows, len(monitor.turns))
    print('Maximum turn-label difference:',
          np.max(np.abs(file_turns - monitor.turns)))
    print('Maximum mean/rms differences [m]:',
          np.max(np.abs(file_mean - monitor.mean_x)),
          np.max(np.abs(file_sigma - monitor.sigma_x)))
    print('Maximum intensity difference:',
          np.max(np.abs(file_count - monitor.num_particles)))

########################################
# Inspect the last frame retained in memory
########################################
xt.Table(
    data    = {
        'turn':       monitor.turns,
        'mean_x_m':   monitor.mean_x,
        'sigma_x_m':  monitor.sigma_x,
        'intensity':  monitor.num_particles},
    index   = 'turn').show(maxwidth = 'full')

################################################################################
# Read the accumulated history from disk
################################################################################
with h5py.File(output_path, 'r') as saved:
    all_turns = saved['turns'][...]
    shape     = saved['stats/beam/mean_x'].shape
print('Saved turn labels:', all_turns)
print('Expected turn labels:', np.arange(N_FRAMES * FRAME_TURNS))
print('Saved mean_x shape / in-memory shape:', shape, monitor.mean_x.shape)
print('Saved statistics:', output_path)

# start_new_frame changes the turn interval and clears the same-size arrays.
# Previously saved frames stay on disk as one flat sequence of absolute turns.
