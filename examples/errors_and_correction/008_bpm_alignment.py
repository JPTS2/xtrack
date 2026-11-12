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
# Correction and threading settings
########################################
RCOND           = 1e-3
THREAD_LENGTH   = 300.0                 # Growing correction-window step [m]

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
tw_before   = line.twiss4d()
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
# Reset the steering, then thread with the same BPM readout geometry
################################################################################

########################################
# Clear correction knobs without changing the BPM calibration
########################################
correction.clear_correction_knobs()
tw_reset    = line.twiss4d()
print('Reset maximum physical x/y difference from the initial orbit [m]:',
      np.max(np.abs(tw_reset.x - tw_before.x)),
      np.max(np.abs(tw_reset.y - tw_before.y)))
print('Correction kicks after reset, horizontal / vertical:',
      correction.x_correction.get_kick_values(),
      correction.y_correction.get_kick_values())

########################################
# Thread an on-axis injected pass through the displaced BPM frames
########################################
threader    = correction.thread(ds_thread = THREAD_LENGTH, rcond_long = RCOND)
start, end  = line.element_names[0], line.element_names[-1]
tw_thread   = line.twiss4d(
    start   = start,
    end     = end,
    init    = tw_before.get_twiss_init(at_element = start),
)
print('Supplied BPM calibration retained after reset / threading:',
      correction.monitor_alignment == alignment,
      threader.monitor_alignment == alignment)
# Threading starts from the original on-axis entrance conditions. Its injected
# trajectory is a different boundary-value problem from periodic correction.
# Both steer against the same displaced and rolled readout frames.

################################################################################
# Distinguish physical orbit from instrument readings
################################################################################

########################################
# Apply the same local-coordinate convention used by the correction
########################################
snapshots   = {
    'Periodic correction':  tw_after,
    'After reset':          tw_reset,
    'Threaded pass':        tw_thread,
}
readings    = {}
for label, tw in snapshots.items():
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

    table = xt.Table(
        data    = {
            'name':     np.asarray(bpms),
            'x_line':   physical.x,
            'y_line':   physical.y,
            'x_bpm':    x_bpm,
            'y_bpm':    y_bpm,
        },
        index   = 'name',
    )
    print(label, 'physical coordinates and local BPM readings [m]:')
    table.show(maxwidth = 'full')
# Nine correctors per plane serve fourteen BPMs. Residual readings need not
# vanish, and reduced readings need not mean a smaller physical orbit.

########################################
# Compare both coordinate frames at the same BPMs
########################################
fig, axes   = plt.subplots(
    nrows   = 2,
    ncols   = 3,
    sharex  = 'col',
    sharey  = 'row',
    figsize = (13, 6),
)
for column, (label, tw) in enumerate(snapshots.items()):
    for row, plane in enumerate(['x', 'y']):
        axes[row, column].plot(tw.rows[bpms].s, tw.rows[bpms][plane] * 1e6,
                                'o-', label = 'Physical coordinates')
        axes[row, column].plot(tw.rows[bpms].s, readings[label][plane] * 1e6,
                                'o--', label = 'Local BPM reading')
        axes[row, column].axhline(0, color = 'black', linewidth = 0.5)
        axes[row, column].grid()
    axes[0, column].set_title(label)
    axes[1, column].set_xlabel('s [m]')
axes[0, 0].set_ylabel('x [µm]')
axes[1, 0].set_ylabel('y [µm]')
axes[0, 0].legend(fontsize = 8)
fig.tight_layout()
plt.show()
