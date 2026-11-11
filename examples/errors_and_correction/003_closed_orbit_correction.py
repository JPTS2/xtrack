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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SHIFT_RMS       = 20e-6                 # Arc quadrupole alignment rms [m]
SEED            = 12345
RESPONSE_CUTOFF = 1e-3                  # Relative singular-value cutoff

################################################################################
# Prepare the nominal response and alignment errors
################################################################################

########################################
# Use the shared ring's named BPMs and correctors
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'      # RF voltage is zero.
tw_ref      = line.twiss4d()
bpms        = line.steering_monitors_x
print('BPMs:', bpms)
print('Correctors:', line.steering_correctors_x)

########################################
# Misalign arc quadrupoles without changing the reference response
########################################
quads   = line.get_table().rows['arc[.]s.*[.]q[fd].*'].name
rng     = np.random.default_rng(SEED)
for name in quads:
    line[name].shift_x  = rng.normal(scale = SHIFT_RMS)
    line[name].shift_y  = rng.normal(scale = SHIFT_RMS)
tw_before   = line.twiss4d()

################################################################################
# Correct the periodic orbit using the nominal uncoupled response
################################################################################
# Reject weak response modes rather than demanding very large steering kicks.
correction  = line.correct_trajectory(
    twiss_table = tw_ref,
    rcond       = RESPONSE_CUTOFF,
)
tw_after    = line.twiss4d()
# This minimizes ideal BPM readings, not every point in the ring. Radiation,
# BPM offsets, measurement noise and corrector limits are absent here.
for plane in ['x', 'y']:
    before  = tw_before.rows[bpms][plane]
    after   = tw_after.rows[bpms][plane]
    print(f'{plane} BPM rms [m]: {np.sqrt(np.mean(before**2)):.3e}'
          f' -> {np.sqrt(np.mean(after**2)):.3e}')

################################################################################
# Compare the orbit and the applied steering kicks
################################################################################

########################################
# Continuous closed orbit and discrete BPM readings
########################################
fig, axes   = plt.subplots(4, 1, sharex = True, figsize = (9, 9))
for axis, plane in zip(axes[:2], ['x', 'y']):
    axis.plot(tw_before.s, tw_before[plane] * 1e3, label = 'Before')
    axis.plot(tw_after.s, tw_after[plane] * 1e3, label = 'After')
    axis.plot(
        tw_after.rows[bpms].s, tw_after.rows[bpms][plane] * 1e3,
        'o',
        label   = 'Corrected BPM readings',
    )
    axis.set_ylabel(f'{plane} [mm]')
    axis.legend()

########################################
# Positive values are positive px and py kicks
########################################
for axis, plane_correction, plane in zip(
        axes[2:], [correction.x_correction, correction.y_correction],
        ['x', 'y']):
    axis.stem(
        plane_correction.s_correctors,
        plane_correction.get_kick_values() * 1e6)
    axis.set_ylabel(f'{plane} kick [µrad]')
axes[-1].set_xlabel('s [m]')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
