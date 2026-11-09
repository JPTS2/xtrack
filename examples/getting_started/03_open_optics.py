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
LATTICE_FILE    = 'test_data/example_lattices/transport/lattice.py'
LINE_NAME       = 'transport'

########################################
# Entrance optics
########################################
BETX_START      = 1.0                   # [m]
BETY_START      = 1.0                   # [m]
ALFX_START      = 0.0
ALFY_START      = 0.0

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Calculate open-line optics
################################################################################
# Unspecified entrance coordinates and dispersion components default to zero.
tw = line.twiss(
    method = '4d',
    betx   = BETX_START,
    bety   = BETY_START,
    alfx   = ALFX_START,
    alfy   = ALFY_START,
)

################################################################################
# Inspect the optics
################################################################################
tw.cols['name s betx bety alfx alfy dx'].show(maxrows = 12)

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
fig.suptitle('Transport line: optics from specified entrance conditions')
fig.tight_layout()
plt.show()
