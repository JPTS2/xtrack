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
SEED            = 12345
SHIFT_RMS       = 20e-6                 # Arc-quadrupole offset RMS [m]
KICK_LIMIT      = 0.5e-6                # Symmetric correction limit [rad]
RESPONSE_CUTOFF = 1e-3                  # Unbounded-response cutoff

################################################################################
# Prepare an erroneous orbit and an unbounded correction for comparison
################################################################################

########################################
# Keep the ideal response before applying alignment errors
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'
reference   = line.twiss4d()
bpms        = line.steering_monitors_x
rng         = np.random.default_rng(SEED)
for name in line.get_table().rows['arc[.].*[.]qd'].name:
    line[name].shift_x  = rng.normal(scale = SHIFT_RMS)
    line[name].shift_y  = rng.normal(scale = SHIFT_RMS)
before      = line.twiss4d()

########################################
# Solve without hardware bounds, then remove that correction
########################################
unbounded   = line.correct_trajectory(
    twiss_table = reference,
    rcond       = RESPONSE_CUTOFF,
)
free_orbit  = line.twiss4d()
free_kicks  = unbounded.x_correction.get_kick_values().copy()
unbounded.clear_correction_knobs()

################################################################################
# Apply native bounds to the correction knobs
################################################################################

########################################
# The same scalar bounds apply to each corrector in each plane
########################################
limits      = (-KICK_LIMIT, KICK_LIMIT)
bounded     = line.correct_trajectory(
    twiss_table         = reference,
    corrector_limits_x  = limits,
    corrector_limits_y  = limits,
    run                 = False,
)
# Limits constrain accumulated correction kicks, not a fresh allowance on
# every call. Existing independent steering settings are zero in this model.
# The bounded solver minimizes the BPM residual subject to these limits.

########################################
# A second call retains the same accumulated bounds
########################################
for iteration in range(2):
    bounded.correct(n_iter = 1)
    for plane in ['x', 'y']:
        kicks   = getattr(bounded, plane + '_correction').get_kick_values()
        print(f'Call {iteration + 1}, {plane}: '
              f'maximum absolute correction = {np.max(np.abs(kicks)):.3e}')
bounded_orbit   = line.twiss4d()
for label, tw in [('Before', before), ('Unbounded', free_orbit),
                  ('Bounded', bounded_orbit)]:
    print(label, 'BPM RMS x/y [m]:',
          np.sqrt(np.mean(tw['x', bpms]**2)),
          np.sqrt(np.mean(tw['y', bpms]**2)))

################################################################################
# Compare residual orbit and hardware demand
################################################################################
fig, axes   = plt.subplots(3, 1, figsize = (9, 8))
for axis, plane in zip(axes[:2], ['x', 'y']):
    for label, tw in [('Before', before), ('Unbounded', free_orbit),
                      ('Bounded', bounded_orbit)]:
        axis.plot(tw['s', bpms], 1e6 * tw[plane, bpms], 'o-', label = label)
    axis.set_ylabel(plane + ' BPM position [µm]')
    axis.set_xlabel('s [m]')
positions   = bounded.x_correction.s_correctors
axes[2].plot(positions, 1e6 * free_kicks, 'o', label = 'Unbounded')
axes[2].plot(
    positions, 1e6 * bounded.x_correction.get_kick_values(), 's',
    label   = 'Bounded',
)
for sign in [-1, 1]:
    axes[2].axhline(sign * KICK_LIMIT * 1e6, color = 'grey', linestyle = '--')
axes[2].set_ylabel('Horizontal correction [µrad]')
axes[2].set_xlabel('s [m]; dashed lines: hardware bounds')
for axis in axes:
    axis.grid()
    axis.legend()
fig.tight_layout()
plt.show()
