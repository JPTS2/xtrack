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
# Model
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'
OBSERVATION     = 'obs.0'

########################################
# Momentum offsets
########################################
DELTA           = 1.0e-5                 # (p - p0) / p0, dimensionless

################################################################################
# Load the ring
################################################################################
# The model supplies an electron reference at p0*c = 1 GeV and a 40 m ring.
# There is no RF or radiation. Periodic 4D optics fixes the momentum offset.
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Calculate periodic closed orbits at fixed momentum
################################################################################

########################################
# Keep the on-momentum solution
########################################
nominal     = line.twiss4d()

########################################
# Solve at positive and negative momentum offsets
########################################
plus        = line.twiss4d(delta0 = DELTA)
minus       = line.twiss4d(delta0 = -DELTA)

################################################################################
# Compare the finite difference with the periodic dispersion
################################################################################

########################################
# Differentiate the closed-orbit coordinates
########################################
# Dispersion differentiates the closed orbit with respect to delta at delta=0.
# px, py are canonical transverse momenta normalized to the reference p0;
# dpx, dpy therefore differentiate those coordinates, not geometrical slopes.
for coordinate, dispersion in [('x', 'dx'), ('px', 'dpx'),
                               ('y', 'dy'), ('py', 'dpy')]:
    finite_difference   = (plus[coordinate] - minus[coordinate]) / (2 * DELTA)
    error               = finite_difference - nominal[dispersion]
    units               = 'm' if coordinate in ['x', 'y'] else 'dimensionless'
    print(dispersion, 'maximum centered-difference error:',
          np.max(np.abs(error)), units)

########################################
# Inspect the displacement at one observation point
########################################
print('Closed orbits and first-order horizontal prediction [m]:')
xt.Table(
    data    = {
        'delta':      np.array([-DELTA, 0, DELTA]),
        'x_orbit':    np.array([minus['x', OBSERVATION],
                                nominal['x', OBSERVATION],
                                plus['x', OBSERVATION]]),
        'x_linear':   nominal['x', OBSERVATION]
                      + nominal['dx', OBSERVATION]
                      * np.array([-DELTA, 0, DELTA])},
    index   = 'delta').show(maxwidth = 'full')

################################################################################
# Plot the full-ring orbit prediction and its residual
################################################################################
# x(delta) = x(0) + Dx*delta + O(delta^2). The centered difference above cancels
# this quadratic term, while the one-sided orbit residual below retains it.
# Numerical orbit and derivative precision also contribute to the residual.
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 7))
for off_momentum, delta, color in [(plus, DELTA, 'C0'),
                                  (minus, -DELTA, 'C1')]:
    predicted_x = nominal.x + nominal.dx * delta
    label       = f'delta = {delta:+.0e}'
    axes[0].plot(
        off_momentum.s, 1e3 * off_momentum.x,
        color   = color,
        label   = label + ' closed orbit',
    )
    axes[0].plot(
        nominal.s, 1e3 * predicted_x,
        color       = color,
        linestyle   = '--',
        label       = label + ' linear',
    )
    axes[1].plot(
        nominal.s, 1e9 * (off_momentum.x - predicted_x),
        color   = color,
        label   = label,
    )
axes[0].set_ylabel('x [mm]')
axes[0].legend()
axes[1].set_ylabel('x - (x0 + Dx delta) [nm]')
axes[1].set_xlabel('s [m]')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
