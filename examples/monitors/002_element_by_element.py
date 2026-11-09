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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Transverse beam
########################################
SEED            = 12345
N_PARTICLES     = 2048
NEMITT_X        = 1.0e-6                # Normalized rms emittance [m rad]
NEMITT_Y        = 2.0e-6                # Normalized rms emittance [m rad]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Calculate the linear beam envelope
################################################################################

########################################
# Obtain optics at element entrances
########################################
tw = line.twiss(method = '4d')

########################################
# Convert emittances to predicted rms sizes
########################################
# On-momentum, uncoupled beam: sigma_x**2 = beta_x * geometric_emittance_x.
# There is no dispersion contribution because all particles have delta = 0.
beta_gamma  = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
sigma_x     = np.sqrt(tw.betx * NEMITT_X / beta_gamma)
sigma_y     = np.sqrt(tw.bety * NEMITT_Y / beta_gamma)

################################################################################
# Track a transverse Gaussian bunch through one turn
################################################################################

########################################
# Draw the normalized launch
########################################
rng = np.random.default_rng(SEED)
particles = line.build_particles(
    method      = '4d',
    x_norm      = rng.standard_normal(N_PARTICLES),
    px_norm     = rng.standard_normal(N_PARTICLES),
    y_norm      = rng.standard_normal(N_PARTICLES),
    py_norm     = rng.standard_normal(N_PARTICLES),
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y,
    zeta        = 0.0,
    delta       = 0.0,
)

########################################
# Record every element boundary
########################################
line.track(particles, turn_by_turn_monitor = 'ONE_TURN_EBE')
record = line.record_last_track
# Axes are (particle-ID slot, element boundary). The final column is the line
# end, not an additional element. Twiss includes that same final boundary.
print('Recorded boundary count:', record.x.shape[1])
print('Surviving macroparticles:', np.count_nonzero(particles.state > 0))
print('First and last s [m]:', record.s[0, 0], record.s[0, -1])

################################################################################
# Compare tracked beam sizes with Twiss
################################################################################

########################################
# Measure the sample rms sizes
########################################
# These small on-momentum launches survive. Standard deviations subtract the
# sample centroid; finite-sample fluctuations remain, as do nonlinear effects.
sample_sigma_x  = np.std(record.x, axis = 0)
sample_sigma_y  = np.std(record.y, axis = 0)
print('Largest relative rms-size differences (x, y):',
      np.max(np.abs(sample_sigma_x / sigma_x - 1)),
      np.max(np.abs(sample_sigma_y / sigma_y - 1)))

########################################
# Plot both envelopes along the lattice
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, measured, predicted, plane in zip(
    axes, [sample_sigma_x, sample_sigma_y], [sigma_x, sigma_y], ['x', 'y']
):
    ax.plot(record.s[0], measured * 1e3, '.', label = 'Tracked sample')
    ax.plot(tw.s, predicted * 1e3, label = 'Twiss prediction')
    ax.set_ylabel(f'sigma_{plane} [mm]')
    ax.legend()
axes[-1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
