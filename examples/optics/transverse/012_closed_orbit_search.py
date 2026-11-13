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
# Native ring and quadrupole offsets
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
SHIFT_X         = 0.1e-3                # Horizontal quad offset [m]
SHIFT_Y         = 0.02e-3               # Vertical quad offset [m]

########################################
# Search conditions
########################################
DELTA0      = 0.0                    # Fixed relative momentum deviation
GUESS_X     = 0.2e-3                 # Initial horizontal guess [m]
GUESS_Y     = -0.1e-3                # Initial vertical guess [m]
SEARCH_AT   = 'bpm.s4.d'                # Alternative search location

################################################################################
# Build a model with a nonzero closed orbit
################################################################################
repo_root                       = Path(__file__).resolve().parents[3]
env                             = xt.load(repo_root / LATTICE_FILE)
line                            = env[LINE_NAME]
env['arc.s1.c1.qf.in'].shift_x  = SHIFT_X
env['arc.s2.c2.qd'].shift_y     = SHIFT_Y
line.build_tracker()

################################################################################
# Find a transverse fixed point at fixed momentum
################################################################################

########################################
# Supply a first guess and longitudinal constraints
########################################
# RF voltage is zero. delta0 fixes momentum; zeta0 fixes the time gauge;
# only the transverse coordinates must return after one turn.
closed = line.find_closed_orbit(
    co_guess    = {'x': GUESS_X, 'y': GUESS_Y},
    delta0      = DELTA0,
    zeta0       = 0.0,
    spin        = False)

########################################
# Track one turn from the returned point
########################################
tracked = closed.copy()
line.track(tracked, num_turns = 1)
coordinates = ['x', 'px', 'y', 'py']
before      = np.array([getattr(closed, c)[0] for c in coordinates])
after       = np.array([getattr(tracked, c)[0] for c in coordinates])
xt.Table(
    data    = {
        'coordinate':  np.array(coordinates),
        'fixed_point': before,
        'after_turn':  after,
        'difference':  after - before},
    index   = 'coordinate').show(maxwidth = 'full')

# x and y are in metres; px and py are normalized canonical momenta.

################################################################################
# Calculate optics about the orbit and choose another search location
################################################################################

########################################
# Reuse the closed-orbit particle
########################################
tw = line.twiss4d(particle_on_co = closed, delta0 = DELTA0)

########################################
# Perform the search at another element
########################################
# co_search_at moves the fixed-point search; the output table retains the
# original line order. Both calculations describe the same closed orbit.
alternative = line.twiss4d(co_search_at = SEARCH_AT, delta0 = DELTA0)
print('Search-location maximum orbit differences [m]:',
      np.max(np.abs(alternative.x - tw.x)),
      np.max(np.abs(alternative.y - tw.y)))
tw.rows[r'bpm\..*'].cols['name s x px y py'].show(maxwidth = 'full')

################################################################################
# Plot the transverse closed orbit
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, column in zip(axes, ['x', 'y']):
    ax.plot(tw.s, tw[column] * 1e3)
    ax.set_ylabel(f'{column} [mm]')
axes[-1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
