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
SEED            = 74
SHIFT_RMS       = 10e-6                 # Arc-quadrupole offset RMS [m]
THREAD_LENGTH   = 300.0                 # Growing correction-window step [m]
RESPONSE_CUTOFF = 1e-3                  # Relative singular-value cutoff

################################################################################
# Set up ideal optics and errors downstream of the sector correctors
################################################################################

########################################
# Supply a response before applying errors
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'
reference   = line.twiss4d()
correction  = line.correct_trajectory(twiss_table = reference, run = False)
start, end  = line.element_names[0], line.element_names[-1]
launch      = reference.get_twiss_init(at_element = start)

########################################
# Use moderate errors so both boundary conditions can be compared
########################################
rng = np.random.default_rng(SEED)
for sector in range(1, 7):
    name                = f'arc.s{sector}.c4.qd'
    line[name].shift_x  = rng.normal(scale = SHIFT_RMS)
    line[name].shift_y  = rng.normal(scale = SHIFT_RMS)
pass_before = line.twiss4d(start = start, end = end, init = launch)

################################################################################
# Steer successively longer passes, then close the orbit
################################################################################

########################################
# Thread from the ring entrance with an on-axis injection
########################################
# Each window extends the corrected range from the entrance to a new endpoint.
# This does not impose periodic closure at the ring entrance.
threader    = correction.thread(
    ds_thread   = THREAD_LENGTH,
    rcond_long  = RESPONSE_CUTOFF,
)
pass_after  = line.twiss4d(start = start, end = end, init = launch)
closed_threaded = line.twiss4d()

########################################
# Refine using periodic rather than injected boundary conditions
########################################
correction.correct(rcond = RESPONSE_CUTOFF)
closed_after    = line.twiss4d()
for plane in ['x', 'y']:
    monitors    = getattr(line, 'steering_monitors_' + plane)
    for label, tw in [('Threaded', closed_threaded),
                      ('Periodic correction', closed_after)]:
        rms = np.sqrt(np.mean(tw[plane, monitors]**2))
        print(f'{label}, {plane}: closed-orbit BPM RMS = {rms:.3e} m')
# Threading can supply a first pass when a closed-orbit search fails. Here the
# errors are deliberately modest: failure of that search is not required.

################################################################################
# Distinguish injected trajectories from periodic orbits
################################################################################
fig, axes   = plt.subplots(2, 2, sharex = 'col', figsize = (11, 6))
for row, plane in enumerate(['x', 'y']):
    for label, tw in [('Before', pass_before), ('Threaded', pass_after)]:
        axes[row, 0].plot(tw.s, 1e3 * tw[plane], label = label)
    for label, tw in [('After threading', closed_threaded),
                      ('Periodic correction', closed_after)]:
        axes[row, 1].plot(tw.s, 1e3 * tw[plane], label = label)
    for axis in axes[row]:
        axis.set_ylabel(plane + ' [mm]')
        axis.grid()
        axis.legend()
axes[0, 0].set_title('On-axis injected pass')
axes[0, 1].set_title('Periodic orbit')
for axis in axes[1]:
    axis.set_xlabel('s [m]')
fig.tight_layout()
plt.show()
