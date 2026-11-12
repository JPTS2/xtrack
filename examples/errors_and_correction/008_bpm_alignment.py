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
# Ideal magnets, displaced and rolled BPM readout frames
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
BPM_SHIFT_X     = 5e-6                  # BPM centre relative to design [m]
BPM_SHIFT_Y     = -3e-6                 # BPM centre relative to design [m]
BPM_ROLL        = 0.01                  # Rotation about the beam axis [rad]

########################################
# Relative response cutoff
########################################
RCOND           = 1e-3

################################################################################
# Define what the BPMs measure
################################################################################

########################################
# Leave the actual magnet geometry unchanged
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.twiss_default['method']    = '4d'      # RF voltage is zero.
tw_before      = line.twiss4d()
bpms        = line.steering_monitors_x

########################################
# Supply BPM calibration geometry to the correction object
########################################
alignment   = {
    name: {
        'shift_x':      BPM_SHIFT_X,
        'shift_y':      BPM_SHIFT_Y,
        'rot_s_rad':    BPM_ROLL,
    }
    for name in bpms
}
# These offsets are readout centres, not a subtraction that recovers the design
# orbit. The correction tries to centre the beam in the displaced BPM frames.
correction  = line.correct_trajectory(
    twiss_table         = tw_before,
    monitor_alignment   = alignment,
    rcond               = RCOND,
    run                 = False,
)

################################################################################
# Steer against readings in the local BPM frames
################################################################################
correction.correct(n_iter = 3)
tw_after    = line.twiss4d()
# The small BPM roll mixes the measured planes. Iteration uses the nominal
# uncoupled response; this is not a coupled-response correction example.

################################################################################
# Distinguish physical orbit from instrument readings
################################################################################

########################################
# Apply the same local-coordinate convention used by the correction
########################################
readings    = {}
for label, tw in [('Before', tw_before), ('After', tw_after)]:
    physical    = tw.rows[bpms]
    x_shifted   = physical.x - BPM_SHIFT_X
    y_shifted   = physical.y - BPM_SHIFT_Y
    x_bpm       = (x_shifted * np.cos(BPM_ROLL)
                   + y_shifted * np.sin(BPM_ROLL))
    y_bpm       = (-x_shifted * np.sin(BPM_ROLL)
                   + y_shifted * np.cos(BPM_ROLL))
    readings[label] = {'x': x_bpm, 'y': y_bpm}
    print(label, 'BPM reading RMS x / y [m]:',
          np.sqrt(np.mean(x_bpm**2)), np.sqrt(np.mean(y_bpm**2)))

table   = xt.Table(
    data    = {
        'name':     np.asarray(bpms),
        'x_line':   tw_after.rows[bpms].x,
        'y_line':   tw_after.rows[bpms].y,
        'x_bpm':    readings['After']['x'],
        'y_bpm':    readings['After']['y'],
    },
    index   = 'name',
)
print('Corrected physical coordinates and local BPM readings [m]:')
table.show(maxwidth = 'full')
# Nine correctors per plane serve fourteen BPMs. Residual readings need not
# vanish, and reduced readings need not mean a smaller physical orbit.

########################################
# Compare both coordinate frames at the same BPMs
########################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for axis, plane in zip(axes, ['x', 'y']):
    axis.plot(tw_after.rows[bpms].s, tw_after.rows[bpms][plane] * 1e6,
              'o-',
              label = 'Physical orbit after steering')
    for label in ['Before', 'After']:
        axis.plot(tw_after.rows[bpms].s, readings[label][plane] * 1e6,
                  'o--',
                  label = f'Local BPM reading: {label.lower()}')
    axis.axhline(0, color = 'black', linewidth = 0.5)
    axis.set_ylabel(f'{plane} [µm]')
    axis.grid()
    axis.legend()
axes[-1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
