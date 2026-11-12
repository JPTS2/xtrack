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
# Native ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
SURVEY_ORIGIN   = 'bpm.s4.f'              # Entrance receiving the assigned pose

########################################
# Assigned global pose
########################################
X0              = 100.0                  # [m]
Y0              = 0.0                    # [m]
Z0              = 200.0                  # [m]
THETA0          = 0.3                    # Horizontal orientation [rad]
PHI0            = 0.0                    # Elevation [rad]
PSI0            = 0.0                    # Roll [rad]

########################################
# Off-momentum closed orbit
########################################
DELTA0          = 1.0e-4                 # Constant relative momentum offset
ORBIT_SCALE     = 10000                  # Floor-plot displacement magnification

################################################################################
# Place a selected entrance at the assigned pose
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

########################################
# Survey with the default first-element origin
########################################
reference = line.survey()

########################################
# Survey with a different global reference frame
########################################
# element0 assigns the supplied position and angles to this element entrance.
# It does not cycle the line: rows and their longitudinal s remain unchanged.
# Returned angles are unwrapped along the line: theta may differ by 2*pi
# multiples from THETA0 at the selected entrance, with the same orientation.
placed = line.survey(
    element0    = SURVEY_ORIGIN,
    X0          = X0,
    Y0          = Y0,
    Z0          = Z0,
    theta0      = THETA0,
    phi0        = PHI0,
    psi0        = PSI0)

################################################################################
# Inspect the selected entrance and the start of the line
################################################################################

########################################
# Compare global poses
########################################
for label, survey in [
    ('Default origin', reference), ('Assigned origin', placed),
]:
    print(label)
    survey.rows[['arc.s1.c1.qf.in', SURVEY_ORIGIN, '_end_point']].cols[
        'name s X Y Z theta phi psi'].show(maxwidth = 'full')

########################################
# Compare distances in the two placements
########################################
# A rigid change of placement preserves distances between surveyed points.
origin_index        = np.flatnonzero(placed.name == SURVEY_ORIGIN)[0]
reference_distance  = np.linalg.norm(
    reference.XYZ[0] - reference.XYZ[origin_index])
placed_distance     = np.linalg.norm(placed.XYZ[0] - placed.XYZ[origin_index])
print('Entrance to selected BPM, default placement [m]:', reference_distance)
print('Ring entrance to selected BPM, assigned placement [m]:', placed_distance)

################################################################################
# Put the same local closed orbit in both global frames
################################################################################

########################################
# Calculate the orbit at the surveyed entrances
########################################
# The demo ring has RF and radiation off. Its off-momentum orbit is periodic
# in the transverse planes, with a fixed delta; this is a 4D optics solution.
tw = line.twiss(method = '4d', delta0 = DELTA0)
orbit_reference = (reference.XYZ + tw.x[:, None] * reference.ex
                   + tw.y[:, None] * reference.ey)
orbit_placed = (placed.XYZ + tw.x[:, None] * placed.ex
                + tw.y[:, None] * placed.ey)
print('Maximum closed-orbit |x| and |y| [mm]:',
      np.max(np.abs(tw.x)) * 1e3, np.max(np.abs(tw.y)) * 1e3)

########################################
# Compare with the rigid change of global placement
########################################
rotation = (placed.E_matrix[origin_index]
            @ reference.E_matrix[origin_index].T)
orbit_transformed = (
    (orbit_reference - reference.XYZ[origin_index]) @ rotation.T
    + placed.XYZ[origin_index])
print('Maximum global-orbit placement difference [m]:',
      np.max(np.linalg.norm(orbit_placed - orbit_transformed, axis = 1)))

################################################################################
# Plot the assigned placement and entrance tangent
################################################################################
# This default pose is planar. The polyline joins element boundaries; bend
# bodies are curved. The tangent arrow uses the surveyed local +s axis.
tangent = placed.E_matrix[origin_index, :, 2]
fig, ax = plt.subplots(figsize = (7, 6))
ax.plot(placed.Z, placed.X, '.-', label = 'Placed reference path')
# Magnify only displacement from the reference, not the reference geometry.
orbit_display = placed.XYZ + ORBIT_SCALE * (orbit_placed - placed.XYZ)
ax.plot(orbit_display[:, 2], orbit_display[:, 0],
        label = f'Closed-orbit displacement x {ORBIT_SCALE}')
ax.plot(Z0, X0, 'o', label = SURVEY_ORIGIN)
ax.quiver(
    Z0, X0, tangent[2], tangent[0],
    angles          = 'xy',
    scale_units     = 'xy',
    scale           = 0.025,
    label           = 'Entrance tangent')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_aspect('equal', adjustable = 'datalim')
ax.grid()
ax.legend()
fig.tight_layout()

################################################################################
# Inspect the physical closed-orbit displacement
################################################################################
fig_orbit, ax_orbit = plt.subplots(figsize = (8, 4))
ax_orbit.plot(tw.s, tw.x * 1e3, label = 'x')
ax_orbit.plot(tw.s, tw.y * 1e3, label = 'y')
ax_orbit.set_xlabel('Reference path length s [m]')
ax_orbit.set_ylabel('Local closed-orbit displacement [mm]')
ax_orbit.set_title(f'Off-momentum orbit, delta = {DELTA0:g}')
ax_orbit.grid()
ax_orbit.legend()
fig_orbit.tight_layout()
plt.show()
