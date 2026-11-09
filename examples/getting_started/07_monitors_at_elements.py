# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/monitors/

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
# Tracking
########################################
N_TURNS         = 32

########################################
# Particle launch
########################################
X_NORM          = 2.0
NEMITT_X        = 1.0e-6                # Normalized rms emittance [m rad]
NEMITT_Y        = 1.0e-6                # Normalized rms emittance [m rad]

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Install the monitors
################################################################################
monitor_start = xt.ParticlesMonitor(
    start_at_turn = 0, stop_at_turn = N_TURNS, num_particles = 1,
)
monitor_middle = xt.ParticlesMonitor(
    start_at_turn = 0, stop_at_turn = N_TURNS, num_particles = 1,
)
env.elements['monitor_start'] = monitor_start
env.elements['monitor_middle'] = monitor_middle
line.insert([
    env.place('monitor_start', at = 'bpm.s1.f'),
    env.place('monitor_middle', at = 'bpm.s4.f'),
])

################################################################################
# Build and track the particle
################################################################################
particles = line.build_particles(
    method   = '4d',
    x_norm   = X_NORM,
    px_norm  = 0.0,
    y_norm   = 0.0,
    py_norm  = 0.0,
    nemitt_x = NEMITT_X,
    nemitt_y = NEMITT_Y,
    zeta     = 0.0,
    delta    = 0.0,
)
line.track(particles, num_turns = N_TURNS)

################################################################################
# Compare the records
################################################################################
table = line.get_table()
print('Monitor locations:')
monitors = table.rows['monitor_start|monitor_middle']
monitors.cols['name s element_type'].show(maxrows = 12)

fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(
    monitor_start.at_turn[0], monitor_start.x[0] * 1e3,
    label = 'bpm.s1.f (s = 1 m)', marker = '.',
)
ax.plot(
    monitor_middle.at_turn[0], monitor_middle.x[0] * 1e3,
    label = 'bpm.s4.f (s = 751 m)', marker = '.',
)
ax.set_xlabel('Turn')
ax.set_ylabel('x [mm]')
ax.set_title('One particle recorded at two locations')
ax.legend()
fig.tight_layout()
plt.show()
