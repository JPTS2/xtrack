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
# Treat a ring section as a single-pass transport line
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
START           = 'arc.s2.c1.qf.in'
END             = 'straight.s6.end'

########################################
# Alignment errors within the selected section
########################################
SHIFT_RMS       = 20e-6                 # Arc quadrupole alignment rms [m]
SEED            = 12345

################################################################################
# Calculate the nominal single-pass response
################################################################################

########################################
# Take entrance optics from the undisturbed periodic ring
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'
tw_periodic = line.twiss4d()
init        = tw_periodic.get_twiss_init(START)
tw_ref      = line.twiss4d(start = START, end = END, init = init)
# The entrance orbit is zero and stays fixed throughout this lesson. The exit
# is free: there is no periodic orbit search in the selected section.

########################################
# Restrict the BPMs, correctors and errors to the section
########################################
bpms        = [name for name in line.steering_monitors_x
               if name in tw_ref.name]
correctors  = [name for name in line.steering_correctors_x
               if name in tw_ref.name]
quads       = line.get_table().rows[START:END].rows[
    'arc[.]s.*[.]q[fd].*'].name
rng         = np.random.default_rng(SEED)
for name in quads:
    line[name].shift_x  = rng.normal(scale = SHIFT_RMS)
    line[name].shift_y  = rng.normal(scale = SHIFT_RMS)
tw_before   = line.twiss4d(start = START, end = END, init = init)

################################################################################
# Steer the trajectory while keeping the entrance coordinates fixed
################################################################################
correction  = line.correct_trajectory(
    start               = START,
    end                 = END,
    twiss_table         = tw_ref,
    monitor_names_x     = bpms,
    monitor_names_y     = bpms,
    corrector_names_x   = correctors,
    corrector_names_y   = correctors,
)
tw_after    = line.twiss4d(start = START, end = END, init = init)
for plane in ['x', 'y']:
    before  = tw_before.rows[bpms][plane]
    after   = tw_after.rows[bpms][plane]
    print(f'{plane} BPM rms [m]: {np.sqrt(np.mean(before**2)):.3e}'
          f' -> {np.sqrt(np.mean(after**2)):.3e}')

################################################################################
# Inspect the single-pass trajectory and the causal response
################################################################################

########################################
# A downstream corrector cannot repair an upstream BPM reading
########################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for axis, plane in zip(axes, ['x', 'y']):
    axis.plot(tw_before.s, tw_before[plane] * 1e3, label = 'Before')
    axis.plot(tw_after.s, tw_after[plane] * 1e3, label = 'After')
    axis.plot(
        tw_after.rows[bpms].s, tw_after.rows[bpms][plane] * 1e3,
        'o',
        label   = 'Corrected BPM readings',
    )
    for name in correctors:
        axis.axvline(tw_ref['s', name], color = 'grey', alpha = 0.3)
    axis.set_ylabel(f'{plane} [mm]')
    axis.grid()
    axis.legend()
axes[-1].set_xlabel('s [m]; grey lines: correctors')
fig.tight_layout()

########################################
# The open response has zeros for downstream correctors
########################################
response    = correction.x_correction.response_matrix
print('Horizontal response [m/rad]; BPM rows, corrector columns:')
print(np.array2string(response, precision = 2, suppress_small = True))
print('BPM row names:', bpms)
print('Corrector column names:', correctors)
plt.show()
