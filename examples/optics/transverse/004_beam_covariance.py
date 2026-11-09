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
# Native uncoupled ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
OBSERVATION     = 'bpm.s1.d'

########################################
# Beam emittances and independent momentum spread
########################################
NEMITT_X        = 1e-6                    # Normalized rms emittance [m rad]
NEMITT_Y        = 2e-6                    # [m rad]
SIGMA_DELTA     = 1e-5                    # Relative momentum rms, same species

################################################################################
# Calculate the periodic 4D optics
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
tw          = line.twiss4d()

################################################################################
# Obtain betatron covariance from the emittances
################################################################################

########################################
# Use the Twiss covariance API
########################################
# With only transverse emittances supplied, this is the betatron contribution.
# Sigma uses canonical (x, px, y, py, zeta, pzeta) coordinates; px and py are
# normalized momenta, not slopes. Its moments are about the beam centroid.
betatron = tw.get_beam_covariance(nemitt_x = NEMITT_X, nemitt_y = NEMITT_Y)
betatron.rows[[OBSERVATION]].cols[
    'name sigma_x sigma_px Sigma11 Sigma12 Sigma22'].show(maxwidth = 'full')

########################################
# Compare the uncoupled horizontal block with Courant-Snyder optics
########################################
beta_gamma  = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
gemitt_x    = NEMITT_X / beta_gamma
index       = np.flatnonzero(tw.name == OBSERVATION)[0]
beta_x      = tw.betx[index]
alpha_x     = tw.alfx[index]
cs_block    = gemitt_x * np.array([
    [beta_x, -alpha_x], [-alpha_x, (1 + alpha_x**2) / beta_x]])
print('API horizontal covariance at', OBSERVATION, ':')
print(betatron.Sigma[index, :2, :2])
print('Courant-Snyder block (x, px):')
print(cs_block)
# Diagonal units are m^2 and 1; the off-diagonal covariance has units m.

################################################################################
# Add dispersion from an independent momentum distribution
################################################################################

########################################
# Form the projected transverse covariance
########################################
# At first order, u = u_beta + D*delta, for u = (x, px, y, py).
# Independence gives Sigma_projected = Sigma_betatron + sigma_delta^2 D D^T.
# This imposed momentum spread needs no longitudinal emittance or RF match.
dispersion  = np.column_stack([tw.dx, tw.dpx, tw.dy, tw.dpy])
projected   = (
    betatron.Sigma[:, :4, :4]
    + SIGMA_DELTA**2 * dispersion[:, :, None] * dispersion[:, None, :])
sigma_x     = np.sqrt(projected[:, 0, 0])
sigma_y     = np.sqrt(projected[:, 2, 2])

########################################
# Compare betatron and projected sizes
########################################
sizes = xt.Table(
    data    = {
        'name':         tw.name,
        's':            tw.s,
        'sigma_x_beta': betatron.sigma_x,
        'sigma_x_full': sigma_x,
        'sigma_y_beta': betatron.sigma_y,
        'sigma_y_full': sigma_y},
    index   = 'name')
print('Beam sizes [m]:')
sizes.rows[['ip.1', OBSERVATION]].show(maxwidth = 'full')

################################################################################
# Plot the rms beam envelopes
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, beta_size, full_size, plane in [
    (axes[0], betatron.sigma_x, sigma_x, 'x'),
    (axes[1], betatron.sigma_y, sigma_y, 'y'),
]:
    ax.plot(tw.s, beta_size * 1e3, label = 'Betatron')
    ax.plot(tw.s, full_size * 1e3, '--', label = 'Including dispersion')
    ax.set_ylabel(f'sigma_{plane} [mm]')
    ax.legend()
axes[-1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
