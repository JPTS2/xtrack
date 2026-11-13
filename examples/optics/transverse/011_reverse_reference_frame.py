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
# Native ring with visible horizontal and vertical orbit offsets
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
SHIFT_X         = 0.1e-3                # Horizontal quad offset [m]
SHIFT_Y         = 0.02e-3               # Vertical quad offset [m]

########################################
# Compare the same physical points
########################################
MARKERS         = ['ip.1', 'straight.s4.mid']

################################################################################
# Calculate optics in proper and reversed local frames
################################################################################

########################################
# Add two quadrupole offsets to a fresh model load
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
env['arc.s1.c1.qf.in'].shift_x = SHIFT_X
env['arc.s2.c2.qd'].shift_y = SHIFT_Y
env['ks_s2'] = 5e-4                  # Nonzero skew strength for the table check

########################################
# Request the reversed-frame result
########################################
proper      = line.twiss4d(strengths = True)
reverse     = line.twiss4d(reverse = True, strengths = True)
converted   = proper.reverse()
print('Reference frames:', proper.reference_frame, reverse.reference_frame)

# reverse describes the same solution in the opposite local reference frame.
# It does not turn the magnets into a second counter-circulating beam model.

################################################################################
# Compare coordinates at zero-length observation markers
################################################################################

########################################
# Display the frame transformation
########################################
# Reversed element entrances are original exits. At a zero-length marker they
# coincide, so matching its name compares the same physical point.
# s -> L-s; x -> -x; px -> px; y -> y; py -> -py.
# Alpha_x/y change sign; dx changes sign while dpx stays unchanged.
columns = ['s', 'x', 'px', 'y', 'py', 'alfx', 'alfy', 'dx', 'dpx']
signs   = np.array([-1, -1, 1, 1, -1, -1, -1, -1, 1])
for name in MARKERS:
    values      = np.array([proper[column, name] for column in columns])
    expected    = signs * values
    expected[0] += line.get_length()
    print('Frame comparison at:', name)
    xt.Table(
        data    = {
            'quantity': np.array(columns),
            'proper':   values,
            'reverse':  np.array([reverse[c, name] for c in columns]),
            'expected': expected},
        index   = 'quantity').show(maxwidth = 'full')

# Positions, s and dx are in metres; momenta, alpha and dpx are dimensionless.

########################################
# Compare calculation and conversion of an existing table
########################################
for column in ['betx', 'bety', 'x', 'px', 'y', 'py', 'dx', 'dpx']:
    difference  = np.max(np.abs(reverse[column] - converted[column]))
    print(f'Calculation/conversion maximum difference in {column}: '
          f'{difference:.3e}')

################################################################################
# Express both orbit tables in the proper frame for an overlay
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, column, sign in zip(axes, ['x', 'y'], [-1, 1]):
    ax.plot(proper.s, proper[column] * 1e3, label = 'Proper frame')
    ax.plot(
        line.get_length() - reverse.s, sign * reverse[column] * 1e3, 'o',
        label = 'Reversed, converted to proper')
    ax.set_ylabel(f'{column} [mm]')
    ax.legend()
axes[-1].set_xlabel('Proper-frame s [m]')
fig.tight_layout()
plt.show()

################################################################################
# Distinguish reversed strength conventions from changing physical magnets
################################################################################
# The table follows the reversed local frame: even normal integrated
# multipoles and odd skew multipoles change sign; normal quadrupoles do not.
# These are table conventions, not new magnet settings or a second beam model.
for name in ['arc.s1.c1.qf.in', 'arc.s1.c1.b1', 'arc.s2.c2.qd']:
    print('Strengths at:', name)
    for field in ['k0l', 'k1l', 'k2l', 'k1sl']:
        print(field, 'proper/reversed:', proper[field, name],
              reverse[field, name])
