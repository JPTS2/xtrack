# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/optics/transverse/

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
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Calculate periodic optics
################################################################################
# With RF off, the 4D solution fixes delta rather than solving RF motion.
tw = line.twiss(method = '4d', delta0 = DELTA0)

################################################################################
# Inspect the optics
################################################################################

########################################
# Tunes and chromaticities
########################################
print(f'Horizontal tune: {tw.qx:.6f}')
print(f'Vertical tune:   {tw.qy:.6f}')
print(f'Chromaticities: dQx/d(delta) = {tw.dqx:.6f}, '
      f'dQy/d(delta) = {tw.dqy:.6f}')

########################################
# Twiss table
########################################
tw.rows[:12].cols['name s betx bety dx mux muy'].show()

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
