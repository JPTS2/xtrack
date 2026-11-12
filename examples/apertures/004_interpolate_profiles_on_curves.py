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
RADIUS_START            = 0.020         # Entrance radius [m]
RADIUS_END              = 0.012         # Exit radius [m]
N_SAMPLES               = 21
N_PROFILE_POINTS        = 128           # Points per profile polygon
SAMPLE_MARGIN           = 0.05          # Stay inside both end profiles [m]
PIPE_CURVATURE_FACTOR   = 0.98          # Second case / reference curvature

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
s_samples   = s_start + np.linspace(
    SAMPLE_MARGIN, length - SAMPLE_MARGIN, N_SAMPLES)
sections    = line.aperture.get_cross_sections_at_s(s_samples, extents = True)
sections.cols['s min_x max_x min_y max_y'].show(maxwidth = 'full')
# Each cross_section is in its own local transverse survey plane. The pose
# column maps that plane into global coordinates. Do not overlay all polygons
# in one global plane while ignoring the changing poses.

################################################################################
# Change only the chamber curvature and retain the same survey query planes
################################################################################

########################################
# Draw the matched placement before editing its live pipe definition
########################################
fig, axes   = plt.subplots(1, 3, figsize = (15, 5))
line.aperture.plot_floor_projection(
    ax      = axes[0],
    origin  = 'bend.chamber',
    s_range = (0, length),
    aspect  = 'auto',
)
axes[0].set_title('Pipe curvature = reference curvature')

########################################
# Rebuild the native geometry after changing the pipe axis
########################################
pipe.curvature      = PIPE_CURVATURE_FACTOR * curvature
line.aperture.build_model()
different_sections  = line.aperture.get_cross_sections_at_s(
    s_samples, extents = True)
print('Reference / changed pipe curvature [m^-1]:', curvature, pipe.curvature)
different_sections.cols['s min_x max_x min_y max_y'].show(maxwidth = 'full')
print('Maximum change in queried survey poses:',
      np.max(np.abs(different_sections.pose - sections.pose)))
# Editing the chamber does not move the reference trajectory. Its centres
# and transverse axes now differ from those of the survey along the bend.
# Interior queries avoid incomplete cuts at the tilted terminal profile:
# the chamber wall ends at that profile; an end cap is not supplied.
line.aperture.plot_floor_projection(
    ax      = axes[1],
    origin  = 'bend.chamber',
    s_range = (0, length),
    aspect  = 'auto',
)
axes[1].set_title(f'Pipe curvature = {PIPE_CURVATURE_FACTOR:g} × reference')
# Both floor projections use unequal scales to resolve widths and curvature.

########################################
# Compare matched and mismatched cuts in the same local planes
########################################
for color, index in enumerate([0, N_SAMPLES // 2, N_SAMPLES - 1]):
    for label, table, style in [
        ('Matched', sections, '--'),
        ('Different', different_sections, '-'),
    ]:
        polygon = table.cross_section[index]
        axes[2].plot(polygon[:, 0] * 1e3, polygon[:, 1] * 1e3, style,
                      color = f'C{color}',
                      label = f'{label}, s = {s_samples[index] - s_start:g} m')
axes[2].set_xlabel('Local survey x [mm]')
axes[2].set_ylabel('Local survey y [mm]')
axes[2].set_title('Each pair uses the same survey plane')
axes[2].set_aspect('equal')
axes[2].grid()
fig.legend(*axes[2].get_legend_handles_labels(),
           loc = 'lower center', ncol = 3, fontsize = 8)
fig.tight_layout(rect = (0, 0.13, 1, 1))
plt.show()
