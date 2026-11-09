# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
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
# Launch and observation
########################################
N_TURNS         = 16
FOLLOWUP_TURNS  = 8
X               = 1.0e-4                # [m]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Record the first tracking interval
################################################################################
particles = line.build_particles(x = X, delta = 0.0)
line.track(
    particles,
    num_turns               = N_TURNS,
    turn_by_turn_monitor    = True,
)
first = line.record_last_track

################################################################################
# Continue without collecting more samples
################################################################################
# No new launch: at_turn continues from the previous call. The saved record is
# a fixed set of samples; it does not grow when tracking continues unmonitored.
line.track(particles, num_turns = FOLLOWUP_TURNS)

################################################################################
# Relate recorded samples to the final particle state
################################################################################

########################################
# Inspect absolute turn labels
########################################
# Rows are particle-ID slots. Each column samples the entrance BEFORE that
# turn's first element. After further unmonitored tracking, the final state
# is later than every sample in the saved record.
print('First record turns:', first.at_turn[0])
print('First sampled x and launch x [m]:', first.x[0, 0], X)
print('Final particle turn and x [m]:', particles.at_turn, particles.x)

########################################
# Plot the recorded interval and the later final state
########################################
fig, ax = plt.subplots(figsize = (8, 4))
valid = first.state[0] > 0
ax.plot(
    first.at_turn[0, valid], first.x[0, valid] * 1e3,
    label   = 'Recorded entrance samples',
    marker  = '.',
)
ax.plot(
    particles.at_turn, particles.x * 1e3,
    label       = 'Later final state (unmonitored interval)',
    marker      = 'o',
    linestyle   = 'none',
)
ax.set_xlabel('Absolute turn at ring entrance')
ax.set_ylabel('x of particle ID 0 [mm]')
ax.legend()
fig.tight_layout()
plt.show()
