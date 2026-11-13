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
BETX_START      = 10.0                   # [m]
BETY_START      = 2.0                   # [m]
ALFX_START      = -2.2
ALFY_START      = 0.5

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Calculate open-line optics
################################################################################
# Rounded entrance conditions give a nearly repeating beta envelope.
# No periodic boundary condition is imposed.
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
print('Open-line optics\n----------------')
print('s, betx, bety, dx [m]; alfx, alfy dimensionless')
tw.cols['name s betx bety alfx alfy dx'].show()

################################################################################
# Plot the result
################################################################################
tw.plot(yl = 'betx bety', yr = 'dx')
plt.show()
