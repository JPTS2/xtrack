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
# Ring and acquisition
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
OBSERVATION     = 'bpm.s1.d'
N_TURNS         = 8
N_PARTICLES     = 512
SEED            = 731

########################################
# Prepared population, not an RF equilibrium
########################################
NEMITT_X        = 1e-6                  # Normalized rms emittance scale [m]
NEMITT_Y        = 2e-6                  # [m]
SIGMA_DELTA     = 1e-5                  # Relative momentum scale
SIGMA_ZETA      = 0.01                  # Independent longitudinal scale [m]
TOTAL_WEIGHT    = 1e10                  # Represented particle population

################################################################################
# Prepare a finite sample from the local uncoupled optics
################################################################################

########################################
# Choose a dispersed read plane
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.cycle(name_first_element = OBSERVATION)
tw          = line.twiss4d()
# The reference is a 1 GeV/c electron; RF voltage and radiation are off.

########################################
# Generate Gaussian normalized coordinates and independent delta/zeta
########################################
rng         = np.random.default_rng(SEED)
normalized  = rng.normal(size = (4, N_PARTICLES))
weights     = rng.uniform(1, 2, N_PARTICLES)
weights    *= TOTAL_WEIGHT / np.sum(weights)
delta       = rng.normal(scale = SIGMA_DELTA, size = N_PARTICLES)
particles   = line.build_particles(
    method      = '4d',
    x_norm      = normalized[0],
    px_norm     = normalized[1],
    y_norm      = normalized[2],
    py_norm     = normalized[3],
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
    delta       = delta,
    zeta        = rng.normal(scale = SIGMA_ZETA, size = N_PARTICLES),
    weight      = weights,
)
initial = particles.copy()

# This finite Gaussian sample approximates the prescribed transverse optics.
# Its independent longitudinal coordinates are not an RF-matched bunch.

################################################################################
# Acquire weighted statistics and covariance-derived optics
################################################################################
monitor = xt.BeamStatsMonitor(
    start_at_turn   = 0,
    stop_at_turn    = N_TURNS,
    stats           = [
        'num_particles', 'mean_x', 'sigma_x', 'betx', 'bety', 'alfx', 'alfy',
        'dx', 'dpx'])
line.insert('beam.statistics', monitor, at = 0)
line.track(particles, num_turns = N_TURNS)
print('Levels/default/shape:', monitor.available_levels,
      monitor.default_level, monitor.mean_x.shape)
print('Recorded turns:', monitor.turns)

################################################################################
# Inspect population moments and distinguish fitted optics from model optics
################################################################################

########################################
# Check the first acquisition against the actual weighted sample
########################################
mean_x  = np.average(initial.x, weights = initial.weight)
sigma_x = np.sqrt(np.average((initial.x - mean_x)**2, weights = initial.weight))
print('Represented population, not macroparticle count:',
      monitor.get('num_particles', turn = 0), N_PARTICLES)
print('First mean_x: monitor / NumPy [m]:', monitor.mean_x[0], mean_x)
print('First sigma_x: monitor / NumPy [m]:', monitor.sigma_x[0], sigma_x)

########################################
# Compare finite-population covariance optics with the preparation model
########################################
# These diagnostics are inferred from stored 6D population moments. They fit
# this finite sample; equality with the lattice Twiss is not imposed.
print('Quantity; first covariance estimate; model value (beta,D in m):')
for quantity in ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']:
    print(quantity, monitor.get(quantity, turn = 0), tw[quantity][0])
print('Final particle states:',
      np.unique(particles.state, return_counts = True))

################################################################################
# Plot the measured centroid and rms width
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].plot(monitor.turns, 1e3 * monitor.mean_x, 'o-')
axes[1].plot(monitor.turns, 1e3 * monitor.sigma_x, 'o-')
axes[0].set_ylabel('Weighted mean x [mm]')
axes[1].set_ylabel('Weighted rms x [mm]')
axes[1].set_xlabel('Turn at the monitor')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
