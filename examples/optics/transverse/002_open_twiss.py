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
LATTICE_FILE    = 'test_data/example_lattices/transport/lattice.py'
LINE_NAME       = 'transport'

########################################
# Entrance optics
########################################
BETX_START  = 10.0                      # [m]
BETY_START  = 2.0                       # [m]
ALFX_START  = -2.2
ALFY_START  = 0.5
DX_START    = 0.0                       # Horizontal dispersion [m]
DPX_START   = 0.0                       # d(px)/d(delta)

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Calculate open-line optics
################################################################################
# The entrance beta/alpha are near the cell-periodic values, so focusing
# remains visible without a large imposed envelope mismatch.
# No periodicity is imposed. Unspecified entrance orbit and vertical dispersion
# components are zero; delta is zero and there is no RF.
tw = line.twiss4d(
    betx    = BETX_START,
    bety    = BETY_START,
    alfx    = ALFX_START,
    alfy    = ALFY_START,
    dx      = DX_START,
    dpx     = DPX_START,
)

################################################################################
# Inspect the optics
################################################################################

########################################
# Compare the entrance and final exit
########################################
print('Inspect the optics\n------------------')
tw.rows[[line.element_names[0], '_end_point']].cols[
    'name s betx alfx bety alfy dx dpx'].show(maxwidth = 'full')

########################################
# Inspect intermediate element entrances
########################################
# _end_point is the last element's exit, rather than another magnet.
tw.cols['name s betx bety dx mux muy'].show(maxwidth = 'full')

################################################################################
# Plot the result
################################################################################
# Use the native optics view; it includes the element layout.
tw.plot()
plt.show()
