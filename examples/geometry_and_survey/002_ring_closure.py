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
# Model
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Floor plot
########################################
PLOT_LABELS     = ['obs.0', 'obs.4']
FIGURE_SIZE     = (7.0, 6.0)            # [inch]

################################################################################
# Load the reference geometry
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
# Survey follows the reference path defined by element lengths and geometric
# bend angles. It does not track a particle or calculate a closed orbit.
survey      = line.survey()

################################################################################
# Compare longitudinal positions and global coordinates
################################################################################

########################################
# Inspect the line table
########################################
# s_start is accumulated reference-path length [m]; angle is each element's
# geometric bend angle [rad], rather than its global orientation.
table = line.get_table(attr = True)
print('First cell bends: longitudinal positions [m] and bend angles [rad]:')
table.rows[['bf.1', 'bd.1']].cols['name s_start s_end angle'].show()

########################################
# Inspect reference frames around the ring
########################################
# Survey rows give element entrances, followed by the final exit row.
# X, Y, Z are global coordinates [m]; theta, phi, psi are orientation angles
# [rad]. With the default initial frame, the entrance tangent points along +Z.
print('Reference geometry: coordinates [m], orientation angles [rad]:')
survey.rows[['obs.0', 'obs.4', '_end_point']].cols[
    'name s X Y Z theta phi psi'
].show(maxwidth = 'full')

################################################################################
# Check closure of position and orientation
################################################################################

########################################
# Compare the entrance and final exit
########################################
position_error  = np.linalg.norm(survey.XYZ[-1] - survey.XYZ[0])
# Rotation matrices avoid identifying orientation by unwrapped angles alone.
frame_error = np.max(np.abs(survey.E_matrix[-1] - survey.E_matrix[0]))
print(f'Reference-path length: {survey.s[-1]:.6f} m')
print(f'Position closure error: {position_error:.3e} m')
print(f'Maximum rotation-matrix difference: {frame_error:.3e}')

########################################
# Interpret the accumulated horizontal rotation
########################################
# In this planar ring positive bend angles decrease the survey theta angle.
# The final theta differs by -2*pi, which describes the same orientation.
theta_change    = survey.theta[-1] - survey.theta[0]
theta_wrapped   = np.arctan2(np.sin(theta_change), np.cos(theta_change))
print(f'Sum of geometric bend angles: {np.sum(table.angle):+.9f} rad')
print(f'Unwrapped theta change: {theta_change:+.9f} rad')
print(f'Theta change modulo 2*pi: {theta_wrapped:+.3e} rad')

################################################################################
# Plot the element-boundary polyline in the horizontal plane
################################################################################
# Connect surveyed entrance positions and the final exit. Bend segments are
# drawn as chords between boundaries; the survey itself uses curved geometry.
fig, ax = plt.subplots(figsize = FIGURE_SIZE)
ax.plot(survey.Z, survey.X, '.-', label = 'Element-boundary polyline')
for name in PLOT_LABELS:
    ax.annotate(
        name,
        xy          = (survey['Z', name], survey['X', name]),
        xytext      = (8, 8),
        textcoords  = 'offset points',
    )
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_aspect('equal', adjustable = 'datalim')
ax.grid()
ax.legend()
ax.set_title('Reference geometry: element-boundary polyline (Z-X)')
fig.tight_layout()
plt.show()
