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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

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
print('Tunes and chromaticities\n------------------------')
print(f'Horizontal tune: {tw.qx:.6f}')
print(f'Vertical tune:   {tw.qy:.6f}')
print(f'Chromaticities: dQx/d(delta) = {tw.dqx:.6f}, '
      f'dQy/d(delta) = {tw.dqy:.6f}')

########################################
# Twiss table
########################################
print('\nFirst 12 Twiss rows\n-------------------')
print('s, betx, bety, dx [m]; mux, muy [turns]')
tw.rows[:12].cols['name s betx bety dx mux muy'].show(maxwidth = 'full')

################################################################################
# Plot the result
################################################################################
tw.plot(yl = 'betx bety', yr = 'dx')
plt.show()
