# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Field-free entrance state
########################################
P0C                 = 1.0e9             # Proton reference p0*c [eV]
X_START             = 2.0e-3            # [m]
Y_START             = -0.5e-3           # [m]
PX_START            = 1.0e-3            # P_x / p0
PY_START            = -0.2e-3           # P_y / p0
ZETA_START          = 0.03              # [m]
DELTA_START         = 0.01              # (p - p0) / p0

########################################
# Frame offsets and rotations
########################################
SHIFT_X             = 0.3e-3            # [m]
SHIFT_Y             = -0.2e-3           # [m]
SHIFT_S             = 0.05              # Displaced observation plane [m]
ROLL                = 0.2               # About s [rad]
PITCH               = -0.015            # About x [rad]
YAW                 = 0.01              # About y [rad]
ROTATION_SEQUENCE   = 'yxs'

########################################
# Magnet for the misalignment comparison
########################################
QUAD_LENGTH         = 0.4               # [m]
K1                  = 0.8               # [m^-2]

################################################################################
# Inspect transverse coordinate signs
################################################################################

########################################
# Translate the frame origin
########################################
initial = xt.Particles(
    mass0   = xt.PROTON_MASS_EV,
    p0c     = P0C,
    x       = X_START,
    y       = Y_START,
    px      = PX_START,
    py      = PY_START,
    zeta    = ZETA_START,
    delta   = DELTA_START,
)
translated = initial.copy()
xt.Translation(shift_x = SHIFT_X, shift_y = SHIFT_Y).track(translated)
print('Expected translated x, y [m]:', X_START - SHIFT_X, Y_START - SHIFT_Y)
print('Tracked translated x, y [m]:', translated.x[0], translated.y[0])

########################################
# Roll the transverse coordinate axes
########################################
# Passive coordinates: x_new = c*x + s*y; y_new = -s*x + c*y.
# The two normalized canonical momenta follow the same rotation.
cos_roll    = np.cos(ROLL)
sin_roll    = np.sin(ROLL)
rolled      = initial.copy()
xt.Rotation(rot_s_rad = ROLL).track(rolled)
print('Expected rolled x, y [m]:',
      cos_roll * X_START + sin_roll * Y_START,
      -sin_roll * X_START + cos_roll * Y_START)
print('Tracked rolled x, y [m]:', rolled.x[0], rolled.y[0])

################################################################################
# Change the observation plane and return to the original frame
################################################################################

########################################
# Apply a longitudinal translation and ordered 3D rotation
########################################
# These maps express the same field-free ray in a new frame and plane.
# Changed momentum components here are coordinate effects, not magnetic kicks.
# The default 'yxs' sequence applies y, then x, then s rotations.
frame_line = xt.Line(
    elements = [
        xt.Translation(shift_x = SHIFT_X, shift_y = SHIFT_Y, shift_s = SHIFT_S),
        xt.Rotation(
            rot_s_rad   = ROLL,
            rot_x_rad   = PITCH,
            rot_y_rad   = YAW,
            seq         = ROTATION_SEQUENCE)],
)

# Retain the particle s instead of resetting it at the line endpoint.
frame_line.reset_s_at_end_turn = False
frame_probe = initial.copy()
frame_line.track(frame_probe)
in_new_frame = frame_probe.copy()

########################################
# Undo the changes in reverse order
########################################
# Backtracking reverses both the element order and the rotation sequence.
frame_line.track(frame_probe, backtrack = True)

################################################################################
# Compare explicit frame changes with magnet misalignment attributes
################################################################################

########################################
# Put a quadrupole between forward and inverse transverse transforms
########################################
# Use transverse shifts and roll only in this magnet comparison.
wrapped_line = xt.Line(
    elements        = [
        xt.Translation(shift_x = SHIFT_X, shift_y = SHIFT_Y),
        xt.Rotation(rot_s_rad = ROLL),
        xt.Quadrupole(length = QUAD_LENGTH, k1 = K1),
        xt.Rotation(rot_s_rad = -ROLL),
        xt.Translation(shift_x = -SHIFT_X, shift_y = -SHIFT_Y),
        xt.Marker()],
    element_names   = ['shift_in', 'roll_in', 'quad', 'roll_out',
                       'shift_out', 'end'],
)
explicit = initial.copy()
wrapped_line.track(explicit, ele_stop = 'end')

########################################
# Apply the equivalent attributes and compare with an aligned magnet
########################################
aligned_magnet      = xt.Quadrupole(length = QUAD_LENGTH, k1 = K1)
misaligned_magnet   = xt.Quadrupole(
    length      = QUAD_LENGTH,
    k1          = K1,
    shift_x     = SHIFT_X,
    shift_y     = SHIFT_Y,
    rot_s_rad   = ROLL,
)
aligned     = initial.copy()
attributed  = initial.copy()
aligned_magnet.track(aligned)
misaligned_magnet.track(attributed)

################################################################################
# Inspect the resulting coordinates
################################################################################
states = {
    'initial':          initial,
    'translated':       translated,
    'rolled':           rolled,
    'new frame':        in_new_frame,
    'restored':         frame_probe,
    'aligned quad':     aligned,
    'explicit quad':    explicit,
    'attributes quad':  attributed}
table = xt.Table(
    data    = {
        'name':     np.array(list(states)),
        'x_mm':     np.array([p.x[0] * 1e3 for p in states.values()]),
        'y_mm':     np.array([p.y[0] * 1e3 for p in states.values()]),
        'px':       np.array([p.px[0] for p in states.values()]),
        'py':       np.array([p.py[0] for p in states.values()]),
        'zeta_mm':  np.array([p.zeta[0] * 1e3 for p in states.values()]),
        's_m':      np.array([p.s[0] for p in states.values()]),
        'delta':    np.array([p.delta[0] for p in states.values()])},
    index   = 'name')

########################################
# Read the frame changes and their inverse
########################################
table.rows[['initial', 'translated', 'rolled', 'new frame', 'restored']].cols[
    'x_mm y_mm px py'].show()
table.rows[['initial', 'new frame', 'restored']].cols[
    's_m zeta_mm delta'].show()

########################################
# Read the magnet exits in the original frame
########################################
table.rows[['aligned quad', 'explicit quad', 'attributes quad']].cols[
    'x_mm y_mm px py'].show()
