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
# Native collision model
########################################
MODEL_FILE      = 'test_data/hllhc15_thick/lhc_thick_with_knobs.json'
IP_NAME         = 'ip1'
IR_START        = 's.ds.l1.b1'
IR_END          = 'e.ds.r1.b1'
PLOT_HALF_WIDTH = 600.0                  # Distance from IP1 [m]

########################################
# Installed IP1 orbit knobs
########################################
CROSSING        = 150.0                  # Beam-1 angle at IP1 [urad]
SEPARATION      = 0.5                    # Beam-1 displacement at IP1 [mm]

################################################################################
# Inspect the collision optics
################################################################################

########################################
# Load the single-beam native model
########################################
repo_root           = Path(__file__).resolve().parents[3]
line                = xt.load(repo_root / MODEL_FILE)
# This fixture has zero orbit knobs. Set the two used below explicitly.
line['on_x1']       = 0
line['on_sep1']     = 0
# Isolate the orbit bumps from their companion dispersion-correction knobs.
line['on_disp']     = 0
reference           = line.twiss(method = '4d')

########################################
# Check the reference and interaction points
########################################
print(f'Beam-1 proton reference p0*c: '
      f'{line.particle_ref.p0c[0] * 1e-12:.3f} TeV')
print(f'Full tunes: Qx = {reference.qx:.6f}, Qy = {reference.qy:.6f}')
reference.rows[['ip1', 'ip2', 'ip5', 'ip8']].cols[
    's betx bety x px y py'].show(maxwidth = 'full')
print(f'Installed IP1 crossing-plane angle: {line["phi_ir1"]:.1f} deg')
# phi_ir1=0 selects horizontal crossing and vertical separation in this model.

################################################################################
# Apply the installed crossing and separation knobs
################################################################################

########################################
# Change the crossing angle
########################################
# Positive on_x1 requests positive px for beam 1, in microradians.
# px is P_x/p0; at delta=0 it agrees with the slope to first order.
line['on_x1']       = CROSSING
crossing            = line.twiss(method = '4d')

########################################
# Add a transverse separation
########################################
# Positive on_sep1 requests positive y for beam 1, in millimetres.
# This is one beam's offset, not the distance between two tracked beams.
line['on_sep1']     = SEPARATION
separated           = line.twiss(method = '4d')

################################################################################
# Measure the response rather than assuming exact knob calibration
################################################################################

########################################
# Inspect the IP1 coordinates
########################################
# Recompute the closed orbit: the knob setting is a requested calibration.
settings    = [reference, crossing, separated]
labels      = np.array(['reference', 'crossing', 'crossing_and_separation'])
xt.Table(
    data    = {
        'setting':  labels,
        'x_mm':     np.array([tw['x', IP_NAME] * 1e3 for tw in settings]),
        'px_urad':  np.array([tw['px', IP_NAME] * 1e6 for tw in settings]),
        'y_mm':     np.array([tw['y', IP_NAME] * 1e3 for tw in settings]),
        'py_urad':  np.array([tw['py', IP_NAME] * 1e6 for tw in settings]),
    },
    index   = 'setting',
).show(maxwidth = 'full')

########################################
# Check residual orbit outside the insertion
########################################
distance    = reference.s - reference['s', IP_NAME]
ir_start    = reference['s', IR_START]
ir_end      = reference['s', IR_END]
outside     = (reference.s < ir_start) | (reference.s > ir_end)
residual_x  = np.max(np.abs(separated.x[outside] - reference.x[outside]))
residual_y  = np.max(np.abs(separated.y[outside] - reference.y[outside]))
print(f'Outside {IR_START} to {IR_END}:')
print(f'Maximum |x - reference| [um]: {residual_x * 1e6:.3e}')
print(f'Maximum |y - reference| [um]: {residual_y * 1e6:.3e}')

################################################################################
# Plot the orbit bumps around IP1
################################################################################
fig, axes   = plt.subplots(2, 1, figsize = (9, 7), sharex = True)
inside      = np.abs(distance) <= PLOT_HALF_WIDTH

########################################
# Horizontal crossing bump
########################################
axes[0].plot(distance[inside], reference.x[inside] * 1e3,
             label  = 'Reference')
axes[0].plot(distance[inside], crossing.x[inside] * 1e3,
             label  = 'Crossing')
axes[0].plot(distance[inside], separated.x[inside] * 1e3, '--',
             label  = 'Crossing and separation')
axes[0].set_ylabel('Beam-1 x [mm]')
axes[0].legend()

########################################
# Vertical separation bump
########################################
axes[1].plot(distance[inside], reference.y[inside] * 1e3,
             label  = 'Reference')
axes[1].plot(distance[inside], separated.y[inside] * 1e3,
             label  = 'Crossing and separation')
axes[1].set_ylabel('Beam-1 y [mm]')
axes[1].set_xlabel('s - s(IP1) [m]')
axes[1].legend()
fig.tight_layout()
plt.show()
