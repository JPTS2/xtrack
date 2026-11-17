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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

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
# With RF voltage zero, use 4D optics at fixed delta. The closed orbit and
# periodic transverse optics are solved together; no entrance beta is given.
tw = line.twiss4d(delta0 = DELTA0)

################################################################################
# Inspect the optics
################################################################################

########################################
# Periodic boundary conditions
########################################
# Rows are element entrances; _end_point is the final exit. Orbit, beta,
# alpha and dispersion return to their initial values in a periodic solution.
print('Inspect the optics\n------------------')
tw.rows[[line.element_names[0], '_end_point']].cols[
    'name s x px betx alfx bety alfy dx dpx'].show(maxwidth = 'full')

########################################
# Accumulated phase advances
########################################
# mux and muy are in turns and are not reduced modulo one.
print(f'Tunes: Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}')
tw.rows[[line.element_names[0], 'ip.1', '_end_point']].cols[
    'name s mux muy'].show(maxwidth = 'full')

################################################################################
# Plot the result
################################################################################
# Use the native optics view; it includes the element layout.
tw.plot()
plt.show()
