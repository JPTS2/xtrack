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
MODEL_DIR       = 'test_data/example_lattices/fcc_ee_z'
SIGMA_SHIFT     = 5e-6                   # Independent x/y offsets [m]
SEED            = 2026
STRENGTH_FILES    = ['strengths.py', 'strengths_larger_ip_beta.py']
LABELS          = ['Nominal beta*', 'Larger beta*']

################################################################################
# Compare the same errors at two optics settings
################################################################################

########################################
# Load the common model and draw one error realization
########################################
repo_root   = Path(__file__).resolve().parents[3]
model_dir   = repo_root / MODEL_DIR
env         = xt.load(model_dir / 'lattice.json')
line        = env.fccee_p_ring
line.configure_radiation(model = None)
table   = line.get_table()
quads   = table.rows[r'(qd0[abc]|qf1[abcd])[lr]\.[0-3]'].name
rng     = np.random.default_rng(SEED)
shift_x = rng.normal(scale = SIGMA_SHIFT, size = len(quads))
shift_y = rng.normal(scale = SIGMA_SHIFT, size = len(quads))
for name, dx, dy in zip(quads, shift_x, shift_y):
    line[name].shift_x  = dx
    line[name].shift_y  = dy
# These are independent offsets of magnet sections, not coherent girder errors.
print('Final-doublet sections:', len(quads))

########################################
# Strength-file loading preserves the applied offsets
########################################
orbits = []
for filename, label in zip(STRENGTH_FILES, LABELS):
    env.call(model_dir / filename)
    tw = line.twiss4d(only_orbit = True)
    orbits.append(tw)
    print(label)
    for coordinate in ['x', 'y']:
        rms = np.sqrt(np.trapezoid(tw[coordinate]**2, tw.s) / line.get_length())
        print(coordinate, 'distance-weighted orbit RMS [m]:', rms)
# Only the closed orbit is calculated here. A converged orbit does not imply
# stable optics around it, and these errors are too large for some such cases.

################################################################################
# Plot the uncorrected orbit sensitivity
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (10, 6),
                         layout = 'constrained')
for label, tw in zip(LABELS, orbits):
    axes[0].plot(tw.s * 1e-3, tw.x * 1e3, label = label)
    axes[1].plot(tw.s * 1e-3, tw.y * 1e3, label = label)
axes[0].set_ylabel('Closed orbit x [mm]')
axes[1].set_ylabel('Closed orbit y [mm]')
axes[1].set_xlabel('s [km]')
for axis in axes:
    axis.legend()
    axis.grid()
plt.show()
