# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Launch and observation points
########################################
X               = np.array([0.1, 0.2, 0.3]) * 1e-3  # [m]
N_TURNS         = 16
OBS_PATTERN     = r'obs\..*'
PARTICLE_ID     = 1
SELECTED_TURN   = 3

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Select an observation family from the element table
################################################################################
table           = line.get_table()
observations    = table.rows.match(name = OBS_PATTERN).name
print('Selected observation points:', observations)

################################################################################
# Record named locations without inserting elements
################################################################################
initial     = line.build_particles(x = X, delta = 0.0)
particles   = initial.copy()
line.track(
    particles,
    num_turns                   = N_TURNS,
    multi_element_monitor_at    = observations,
)
record = line.record_multi_element_last_track
# get('x') axes are (turn, particle-ID slot, observation point).
print('Full x record shape:', record.get('x').shape)

################################################################################
# Install monitor elements at the same locations
################################################################################

########################################
# Create separate records and place them before the markers
########################################
for name in observations:
    env.elements[f'monitor.{name}'] = xt.ParticlesMonitor(
        start_at_turn   = 0,
        stop_at_turn    = N_TURNS,
        num_particles   = len(X),
    )
line.insert([
    env.place(f'monitor.{name}', at = f'{name}@start')
    for name in observations
])

########################################
# Repeat the same launch
########################################
replayed = initial.copy()
line.track(replayed, num_turns = N_TURNS)

################################################################################
# Compare observations at each physical point
################################################################################

########################################
# Reconcile the two array layouts
########################################
# ParticlesMonitor axes are (particle-ID slot, sample), so transpose its x.
for name in observations:
    installed   = line[f'monitor.{name}']
    difference  = installed.x.T - record.get('x', obs_name = name)
    print(name, 'largest x difference [m]:', np.max(np.abs(difference)))
    print('Recorded s [m]:', installed.s[0, 0])
    print('Monitor element index:', installed.at_element[0, 0])

# The installed record identifies its monitor element; the named record samples
# the original marker entrance. The zero-length monitor does not change motion.

########################################
# Select one particle, turn and location
########################################
locations   = record.get('x', particle_id = PARTICLE_ID, turn = SELECTED_TURN)
value       = record.get(
    'x',
    obs_name    = 'obs.4',
    particle_id = PARTICLE_ID,
    turn        = SELECTED_TURN,
)
print('x at both locations for the selected turn [m]:', locations)
print('One selected sample x [m]:', value)

################################################################################
# Plot the selected particle at both observation points
################################################################################
fig, ax = plt.subplots(figsize = (8, 4))
for name in observations:
    ax.plot(
        np.arange(N_TURNS),
        record.get('x', obs_name = name, particle_id = PARTICLE_ID) * 1e3,
        label   = name,
        marker  = '.',
    )
ax.set_xlabel('Turn')
ax.set_ylabel(f'x of particle ID {PARTICLE_ID} [mm]')
ax.legend()
fig.tight_layout()
plt.show()
