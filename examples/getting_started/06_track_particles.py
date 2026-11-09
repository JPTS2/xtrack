# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/particles/
#                   examples/tracking/ and examples/monitors/

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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Tracking
########################################
N_TURNS         = 64

########################################
# Particle launch
########################################
X_NORM          = np.array([0.5, 1.0, 2.0, 3.0])
Y_NORM          = 0.5

########################################
# Emittances
########################################
NEMITT_X        = 1.0e-6                # Normalized rms emittance [m rad]
NEMITT_Y        = 1.0e-6                # Normalized rms emittance [m rad]

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Build the particles
################################################################################
# Normalized launch coordinates are scaled using the stated emittances and
# the ring's on-momentum 4D optics. These four particles are a launch scan,
# not samples of a Gaussian bunch.
particles = line.build_particles(
    method   = '4d',
    x_norm   = X_NORM,
    px_norm  = 0.0,
    y_norm   = Y_NORM,
    py_norm  = 0.0,
    nemitt_x = NEMITT_X,
    nemitt_y = NEMITT_Y,
    zeta     = 0.0,
    delta    = 0.0,
)

################################################################################
# Track and inspect the record
################################################################################
# Static magnets conserve delta; zeta can evolve without RF confinement.
line.track(
    particles,
    num_turns            = N_TURNS,
    turn_by_turn_monitor = True,
)
record = line.record_last_track

# Rows identify particles; columns identify recorded turns. For this simple
# scan there are no losses or reordering. In general, use particle_id/state
# in the record rather than assuming final particle-array order is unchanged.
print(f'Turn-by-turn x record shape: {record.x.shape}')
print(f'Surviving particles: {np.count_nonzero(particles.state > 0)} '
      f'of {len(X_NORM)}')

################################################################################
# Plot the transverse motion
################################################################################
# All launches have the same vertical amplitude, so their y curves largely
# coincide in this uncoupled lattice.
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for particle_index, amplitude in enumerate(X_NORM):
    alive = record.state[particle_index] > 0
    turns = record.at_turn[particle_index, alive]
    axes[0].plot(
        turns, record.x[particle_index, alive] * 1e3,
        label = f'x_norm = {amplitude:g}', marker = '.', markersize = 3,
    )
    axes[1].plot(turns, record.y[particle_index, alive] * 1e3)

axes[0].set_ylabel('x [mm]')
axes[0].legend(ncol = 2)
axes[1].set_ylabel('y [mm]')
axes[1].set_xlabel('Turn (recorded at the ring entrance)')
fig.suptitle('Transverse tracking at fixed reference momentum')
fig.tight_layout()
plt.show()
