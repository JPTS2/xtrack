# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Launch and recording
########################################
X               = np.array([0.1, 0.2, 0.3]) * 1e-3  # [m]
FIRST_TURN      = 2
STOP_TURN       = 5                     # Exclusive
ID_RANGE        = (1, 3)                # Lower inclusive, upper exclusive
N_TURNS         = 24
N_FRAMES        = 3
FRAME_PERIOD    = 8                     # Turns between window starts

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Record a contiguous window for selected particle IDs
################################################################################

########################################
# Configure the recording window
########################################
initial = line.build_particles(x = X, delta = 0.0)
monitor = xt.ParticlesMonitor(
    start_at_turn       = FIRST_TURN,
    stop_at_turn        = STOP_TURN,
    particle_id_range   = ID_RANGE,
)

########################################
# Track and read absolute turn labels
########################################
particles = initial.copy()
line.track(
    particles,
    num_turns               = N_TURNS,
    turn_by_turn_monitor    = monitor,
)
print('Record shape (ID slot, sample):', monitor.x.shape)
print('Recorded particle IDs:', monitor.particle_id[:, 0])
print('Absolute recorded turns:', monitor.at_turn[0])

################################################################################
# Sample the same window inside a manual turn loop
################################################################################

########################################
# Recreate the configuration without recorded data
########################################
# Monitor dictionary round trips preserve configuration, not trajectories.
manual = xt.ParticlesMonitor.from_dict(monitor.to_dict())
print('Valid samples before filling:', np.sum(manual.state > 0))

########################################
# Control sampling explicitly
########################################
replayed = initial.copy()
for turn in range(N_TURNS):
    # The monitor uses particles.at_turn, not the Python loop counter.
    manual.track(replayed)
    line.track(replayed)
print('Largest manual/automatic x difference [m]:',
      np.max(np.abs(manual.x - monitor.x)))

################################################################################
# Repeat the window at separated turns
################################################################################

########################################
# Configure repeated frames
########################################
frames = xt.ParticlesMonitor(
    start_at_turn       = FIRST_TURN,
    stop_at_turn        = STOP_TURN,
    particle_id_range   = ID_RANGE,
    n_repetitions       = N_FRAMES,
    repetition_period   = FRAME_PERIOD,
)

########################################
# Inspect the extra frame axis
########################################
repeated = initial.copy()
line.track(
    repeated,
    num_turns               = N_TURNS,
    turn_by_turn_monitor    = frames,
)
# Multiple frames add a leading axis; a single frame has no frame axis.
print('Repeated-window record shape:', frames.x.shape)
if N_FRAMES > 1:
    print('Absolute turns in each frame:', frames.at_turn[:, 0, :])
else:
    print('Absolute turns:', frames.at_turn[0])
