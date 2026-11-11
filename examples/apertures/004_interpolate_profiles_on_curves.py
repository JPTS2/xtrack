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
# A bend in the shared ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
BEND_NAME       = 'arc.s1.c1.b1'

########################################
# Circular profiles and sampling
########################################
RADIUS_START        = 0.020                 # Entrance radius [m]
RADIUS_END          = 0.012                 # Exit radius [m]
N_SAMPLES           = 21
N_PROFILE_POINTS    = 128               # Points per profile polygon

################################################################################
# Match the pipe axis to the surveyed bend
################################################################################

########################################
# Read the native bend length and curvature
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
bend        = line[BEND_NAME]
length      = bend.length
curvature   = bend.angle / length
s_start     = line.get_table()['s', BEND_NAME]

########################################
# Define profiles along a curved pipe rather than along its entrance tangent
########################################
entrance    = env.aperture.new_profile(
    'entrance', 'Circle', radius = RADIUS_START)
exit_shape  = env.aperture.new_profile(
    'exit', 'Circle', radius = RADIUS_END)
pipe        = env.aperture.new_pipe(
    'tapered_bend',
    curvature   = curvature,
    positions   = [entrance.place(at = 0), exit_shape.place(at = length)],
)
line.aperture.place_pipe('bend.chamber', pipe, at = BEND_NAME)
line.aperture.num_profile_points    = N_PROFILE_POINTS
line.aperture.build_model()
# Pipe curvature has the survey sign convention. Equal curvature and aligned
# entrance frames make its axis follow this bend's reference trajectory.

################################################################################
# Interpolate cross-sections in the changing local survey frames
################################################################################

########################################
# Obtain native polygons and axis intersections at requested positions
########################################
s_samples   = s_start + np.linspace(0, length, N_SAMPLES)
sections    = line.aperture.get_cross_sections_at_s(s_samples, extents = True)
sections.cols['s min_x max_x min_y max_y'].show(maxwidth = 'full')
# Each cross_section is in its own local transverse survey plane. The pose
# column maps that plane into global coordinates. Do not overlay all polygons
# in one global plane while ignoring the changing poses.

########################################
# Plot the curved placement and several local cross-sections
########################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
line.aperture.plot_floor_projection(
    ax      = axes[0],
    origin  = 'bend.chamber',
    s_range = (0, length),
    aspect  = 'auto',
)
axes[0].set_title('Curved chamber: unequal axis scales')
# Unequal scales resolve both the chamber width and the survey curvature.
for index in [0, N_SAMPLES // 2, N_SAMPLES - 1]:
    polygon = sections.cross_section[index]
    axes[1].plot(
        1e3 * polygon[:, 0], 1e3 * polygon[:, 1],
        label   = f's from bend entrance = {s_samples[index] - s_start:g} m',
    )
axes[1].set_xlabel('Local survey x [mm]')
axes[1].set_ylabel('Local survey y [mm]')
axes[1].set_title('Cross-sections in their own local planes')
axes[1].set_aspect('equal')
axes[1].grid()
axes[1].legend()
fig.tight_layout()
plt.show()
