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
# Monitor locations
########################################
LOCATION_1      = 'bpm.s1.f'
LOCATION_2      = 'bpm.s1.d'

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
monitor_1 = xt.ParticlesMonitor(
    start_at_turn = 0, stop_at_turn = N_TURNS, num_particles = 1,
)
monitor_2 = xt.ParticlesMonitor(
    start_at_turn = 0, stop_at_turn = N_TURNS, num_particles = 1,
)
env.elements['monitor_1'] = monitor_1
env.elements['monitor_2'] = monitor_2
print('Install the monitors\n--------------------')
line.insert([
    env.place('monitor_1', at = LOCATION_1),
    env.place('monitor_2', at = LOCATION_2),
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
# These are two records of the same particle, at different optical phases.
# Passive BPM devices identify locations; ParticlesMonitor records the motion.
tw = line.twiss(method = '4d')
print('\nMonitor locations and optics\n---------------------------')
print('s, betx [m]; mux [turns]')
tw.rows[['monitor_1', 'monitor_2']].cols[
    'name s betx mux'].show(maxwidth = 'full')
print('The same particle is recorded at both locations on every turn.')

################################################################################
# Plot the two records
################################################################################
fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(
    monitor_1.at_turn[0], monitor_1.x[0] * 1e3,
    label = LOCATION_1, marker = '.',
)
ax.plot(
    monitor_2.at_turn[0], monitor_2.x[0] * 1e3,
    label = LOCATION_2, marker = '.',
)
ax.set_xlabel('Turn')
ax.set_ylabel('x [mm]')
ax.set_title('One particle recorded at two locations')
ax.margins(y = 0.20)
ax.legend(loc = 'upper center', ncol = 2)
fig.tight_layout()
plt.show()
