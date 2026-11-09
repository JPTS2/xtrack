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
# Ring and full-turn acquisition grid
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
N_TURNS         = 4
N_PARTICLES     = 128
N_SLICES        = 16

########################################
# Prepared coasting population and initial displacement pattern
########################################
TOTAL_WEIGHT    = 1e10
AMPLITUDE       = 1e-4                 # Horizontal displacement scale [m]
SPATIAL_MODE    = 3                    # Integer cycles across the initial turn

################################################################################
# Load the bare magnetic ring and populate one full longitudinal turn
################################################################################

########################################
# Configure full-turn slicing at the ring entrance
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
length      = line.get_length()
beta0       = float(line.particle_ref.beta0[0])
monitor     = xt.BeamStatsMonitor(
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS,
    coasting        = True,
    num_slices      = N_SLICES,
    stats           = ['num_particles', 'mean_x', 'mean_zeta'],
)
line.insert('coasting.statistics', monitor, at = 0)

########################################
# Use midpoint positions, avoiding slice and effective-turn boundaries
########################################
zeta    = length * ((np.arange(N_PARTICLES) + 0.5) / N_PARTICLES - 0.5)
particles   = line.build_particles(
    x       = AMPLITUDE * np.sin(2 * np.pi * SPATIAL_MODE * zeta / length),
    px      = 0.0,
    zeta    = zeta,
    delta   = 0.0,
    weight  = TOTAL_WEIGHT / N_PARTICLES,
)

# This is a prepared coasting population, with RF voltage and radiation off.
# Its displacement pattern is not asserted to be a collective eigenmode.
line.track(particles, num_turns = N_TURNS)

################################################################################
# Inspect the local arrival clock and full-turn slice arrays
################################################################################

########################################
# Distinguish reference-turn bins from individual revolution counters
########################################
# Arrival in turn units is reference_turns-zeta/C. The record turn is its
# nearest integer; slices order the residual arrival phase from -1/2 to +1/2.
# Outlying zeta can therefore populate neighboring effective-turn records.
print('Levels/default/shape:', monitor.available_levels,
      monitor.default_level, monitor.mean_x.shape)
print('Recorded reference-turn bins:', monitor.turns)
print('Weight by effective turn:', monitor.get('num_particles', level = 'beam'))
print('Coasting zeta_centers property:', monitor.zeta_centers)

########################################
# Convert the same acquisition grid to unwrapped zeta and local arrival time
########################################
zz  = monitor.zeta_centers_unwrapped(line_length = length)
tt  = monitor.time_centers(line_length = length, beta0 = beta0)
print('Unwrapped zeta/time array shapes:', zz.shape, tt.shape)
print('First-row unwrapped zeta centers [m]:', zz[0])

# Time zero is local to this monitor; no upstream flight time is added.
# At fixed reference speed, time=-unwrapped_zeta/(beta0*c).

################################################################################
# Plot populated records against their unwrapped local arrival times
################################################################################
counts  = monitor.num_particles
valid   = counts > 0
fig, ax = plt.subplots(figsize = (9, 4))
ax.plot(1e6 * tt[valid], 1e3 * monitor.mean_x[valid], '.-')
ax.set_xlabel('Unwrapped local arrival time [us]')
ax.set_ylabel('Weighted slice mean x [mm]')
ax.grid()
fig.tight_layout()
plt.show()
