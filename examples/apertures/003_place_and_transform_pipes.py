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
# Short transport line and reusable straight pipe
########################################
LATTICE_FILE    = 'test_data/example_lattices/transport/lattice.py'
PIPE_LENGTH     = 1.0                   # [m]
HALF_X          = 0.020                 # Elliptical horizontal semiaxis [m]
HALF_Y          = 0.010                 # Elliptical vertical semiaxis [m]
PROFILE_POINTS  = 128                   # Polygon points for interpolation

########################################
# Alignment of the second installed pipe, relative to its anchor frame
########################################
SHIFT_X         = 0.004                 # [m]
SHIFT_Y         = -0.002                # [m]
ROLL            = 0.20                  # Rotation about pipe-local z [rad]

################################################################################
# Define one pipe and install it at two survey anchors
################################################################################

########################################
# Reuse the maintained transport-line geometry
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['transport']
profile     = env.aperture.new_profile(
    'ellipse', 'Ellipse',
    a   = HALF_X,
    b   = HALF_Y,
)
pipe        = env.aperture.new_pipe(
    'straight_tube',
    positions   = [profile.place(at = 0.0), profile.place(at = PIPE_LENGTH)],
)

########################################
# Keep the first installation aligned and displace the second one
########################################
line.aperture.place_pipe('aligned', pipe, at = 'qf.1@start')
line.aperture.place_pipe(
    'shifted_and_rolled', pipe,
    at          = 'qd.1@start',
    shift_x     = SHIFT_X,
    shift_y     = SHIFT_Y,
    rot_z_rad   = ROLL,
)
# Pipe placements use local Cartesian (x,y,z), with +z along the pipe axis.
# Profile stations instead use the longitudinal coordinate s. A placement's
# rot_z_rad rolls the whole pipe; it does not rotate the beam-line magnets.

################################################################################
# Build the aperture geometry and inspect the installed cross-sections
################################################################################

########################################
# Build only after all definitions and installations have been added
########################################
line.aperture.num_profile_points    = PROFILE_POINTS
line.aperture.build_model()
line.aperture.get_pipe_table().cols[
    'name pipe_name survey_reference s_start s_end'].show(maxwidth = 'full')
line.aperture.get_bounds_table().cols[
    'name profile_name s s_start s_end'].show(maxwidth = 'full')
print('Shifted pipe placement matrix in its anchor frame:')
print(line.aperture.pipe_positions['shifted_and_rolled'].transformation)
# Both installations refer to the same named pipe definition.

########################################
# Compare sections halfway along the two installed pipes
########################################
table       = line.get_table()
sections_s  = np.array([
    table['s', 'qf.1'] + PIPE_LENGTH / 2,
    table['s', 'qd.1'] + PIPE_LENGTH / 2,
])
sections    = line.aperture.get_cross_sections_at_s(sections_s, extents = True)
sections.cols['s min_x max_x min_y max_y'].show(maxwidth = 'full')
# These cuts are inside the installed pipes. They do not demonstrate the
# chamber assumed across gaps or track particles against a physical wall.

################################################################################
# Plot the global placement and local transverse geometry
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
line.aperture.plot_floor_projection(ax = axes[0], aspect = 'auto')
axes[0].set_title('Installed pipes in the global survey frame')
for label, polygon in zip(['Aligned', 'Shifted and rolled'],
                           sections.cross_section):
    closed  = np.vstack([polygon, polygon[0]])
    axes[1].plot(closed[:, 0] * 1e3, closed[:, 1] * 1e3, label = label)
axes[1].set_xlabel('Local survey x [mm]')
axes[1].set_ylabel('Local survey y [mm]')
axes[1].set_title('Cross-sections in each local survey frame')
axes[1].set_aspect('equal')
axes[1].legend()
for ax in axes:
    ax.grid()
# The floor projection uses unequal axis scales to make the pipe walls visible.
fig.tight_layout()
plt.show()
