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
# Native ring and skew quadrupole components
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'
K1S_QF1         = 0.05                    # Skew k1 at qf.1 [m^-2]
K1S_QD5         = -0.04                   # Skew k1 at qd.5 [m^-2]

########################################
# Independent normal-mode populations
########################################
NEMITT_1        = 1e-6                    # Mode 1 normalized rms emittance [m]
NEMITT_2        = 2e-6                    # Mode 2 normalized rms emittance [m]

################################################################################
# Calculate the uncoupled and coupled optics
################################################################################

########################################
# Keep the original periodic solution
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
uncoupled   = line.twiss4d()

########################################
# Add two skew field components to the loaded model
########################################
# These are gradient components of existing 0.3 m quadrupoles, not integrated
# strengths. Each fresh load is independent of the native model on disk.
env['qf.1'].k1s = K1S_QF1
env['qd.5'].k1s = K1S_QD5
tw = line.twiss4d()

################################################################################
# Inspect the two normal modes in both physical planes
################################################################################

########################################
# Compare mode tunes and projected beta functions
########################################
# Mode labels continue from the weakly coupled horizontal/vertical branches.
# With coupling, qx and qy label modes rather than independent plane motion.
for label, optics in [('Uncoupled', uncoupled), ('Coupled', tw)]:
    print(f'{label}: mode tunes {optics.qx:.9f}, {optics.qy:.9f}')
    optics.rows[['obs.0', 'obs.4']].cols[
        'name s betx1 betx2 bety1 bety2'].show(maxwidth = 'full')

########################################
# Plot each plane's response to each mode
########################################
# betx1 is the horizontal projection of mode 1; betx2 is that of mode 2.
# bety1 and bety2 are their vertical projections. All four are in metres.
fig, axes = plt.subplots(2, 2, sharex = True, figsize = (10, 7))
for ax, column in zip(axes.flat, ['betx1', 'betx2', 'bety1', 'bety2']):
    ax.plot(uncoupled.s, uncoupled[column], '--', label = 'Uncoupled')
    ax.plot(tw.s, tw[column], label = 'Coupled')
    ax.set_ylabel(f'{column} [m]')
    ax.legend()
for ax in axes[-1]:
    ax.set_xlabel('s [m]')
fig.tight_layout()
plt.show()

################################################################################
# Combine mode emittances into projected beam sizes
################################################################################

########################################
# Sum the two mode contributions
########################################
# For independent mode populations at delta = 0, both contribute to each plane.
# In this coupled calculation the API's nemitt_x/y are mode 1/2 emittances.
beta_gamma  = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
gemitt_1    = NEMITT_1 / beta_gamma
gemitt_2    = NEMITT_2 / beta_gamma
sigma_x     = np.sqrt(gemitt_1 * tw.betx1 + gemitt_2 * tw.betx2)
sigma_y     = np.sqrt(gemitt_1 * tw.bety1 + gemitt_2 * tw.bety2)

########################################
# Compare with the full covariance API
########################################
covariance  = tw.get_beam_covariance(nemitt_x = NEMITT_1, nemitt_y = NEMITT_2)
sizes       = xt.Table(
    data    = {
        'name':            tw.name,
        'sigma_x_modes':   sigma_x,
        'sigma_x_api':     covariance.sigma_x,
        'sigma_y_modes':   sigma_y,
        'sigma_y_api':     covariance.sigma_y},
    index   = 'name')
print('Projected rms beam sizes [m]:')
sizes.rows[['obs.0', 'obs.4']].show(maxwidth = 'full')
