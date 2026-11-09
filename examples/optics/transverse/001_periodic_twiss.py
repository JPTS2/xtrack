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
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Momentum offset
########################################
DELTA0          = 0.0                   # (p - p0) / p0, reference species

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Calculate periodic optics
################################################################################
# With no RF, use 4D optics at fixed delta. The closed orbit and periodic
# transverse optical functions are solved together; no entrance beta is given.
tw = line.twiss4d(delta0 = DELTA0)

################################################################################
# Inspect the optics
################################################################################

########################################
# Periodic boundary conditions
########################################
# Rows are element entrances; _end_point is the final exit. Orbit, beta,
# alpha and dispersion return to their initial values in a periodic solution.
tw.rows[['obs.0', '_end_point']].cols[
    'name s x px betx alfx bety alfy dx dpx'].show(maxwidth = 'full')

########################################
# Accumulated phase advances
########################################
# mux and muy are in turns and are not reduced modulo one.
print(f'Tunes: Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}')
tw.rows[['obs.0', 'obs.4', '_end_point']].cols[
    'name s mux muy'].show(maxwidth = 'full')

################################################################################
# Plot the result
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))

########################################
# Beta functions
########################################
axes[0].plot(tw.s, tw.betx, label = r'$\beta_x$')
axes[0].plot(tw.s, tw.bety, label = r'$\beta_y$')
axes[0].set_ylabel(r'$\beta$ [m]')
axes[0].legend()

########################################
# Horizontal dispersion
########################################
axes[1].plot(tw.s, tw.dx)
axes[1].set_ylabel(r'$D_x$ [m]')
axes[1].set_xlabel('s [m]')
fig.suptitle(f'Periodic 4D optics: Qx = {tw.qx:.3f}, Qy = {tw.qy:.3f}')
fig.tight_layout()
plt.show()
