# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# A straight chamber with independently tilted profile stations
########################################
LINE_LENGTH         = 0.22             # Survey length [m]
RADIUS              = 0.025            # Radius in each profile's own plane [m]
PROFILE_STATIONS    = 0.08 + 0.008 * np.arange(8)  # Pipe coordinates [m]
PROFILE_TILTS       = np.deg2rad(np.arange(35, 59, 3))  # About local y [rad]
QUERY_S             = 0.108            # Survey cut position [m]
PROFILE_POINTS      = 128

################################################################################
# Install profile planes with overlapping longitudinal footprints
################################################################################

########################################
# Keep each profile orientation in its own placement
########################################
env         = xt.Environment()
line        = env.new_line(
    components = [env.new('drift', xt.Drift, length = LINE_LENGTH)],
)
profile     = env.aperture.new_profile('round', 'Circle', radius = RADIUS)
pipe        = env.aperture.new_pipe(
    'tilted_stations',
    positions = [
        profile.place(at = position, rot_y_rad = tilt)
        for position, tilt in zip(PROFILE_STATIONS, PROFILE_TILTS)
    ],
)
line.aperture.place_pipe('chamber', pipe, at = 'drift')
line.aperture.num_profile_points = PROFILE_POINTS
line.aperture.build_model()
# Each Circle is round in its own plane. Tilting a station spreads that
# profile over a longitudinal interval; neighbouring intervals can overlap.
# This geometry-only example supplies no beam distribution or tracking limits.

########################################
# Inspect installed centres and their native longitudinal bounds
########################################
bounds          = line.aperture.get_bounds_table()
bounds.cols['name s s_start s_end'].show(maxwidth = 'full')
intersected     = (bounds.s_start <= QUERY_S) & (QUERY_S <= bounds.s_end)
print('Installed profiles whose longitudinal bounds contain the query:',
      np.count_nonzero(intersected), '/', len(bounds))

################################################################################
# Request the combined cross-section in the survey plane
################################################################################
sections    = line.aperture.get_cross_sections_at_s([QUERY_S], extents = True)
sections.cols['s min_x max_x min_y max_y'].show(maxwidth = 'full')
polygon     = sections.cross_section[0]
print('Finite returned polygon vertices / total:',
      np.count_nonzero(np.isfinite(polygon).all(axis = 1)), len(polygon))
# The cut follows corresponding polygon vertices through successive stations.
# Choosing only the profiles whose centre s brackets QUERY_S would omit
# tilted profiles intersected by that plane. The result need not be a circle.

################################################################################
# Relate the combined cut to the installed profile footprints
################################################################################

########################################
# Show which profile planes extend across the query position
########################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
for index in range(len(bounds)):
    color   = 'C1' if intersected[index] else '0.65'
    axes[0].plot([bounds.s_start[index], bounds.s_end[index]],
                  [index, index], color = color, linewidth = 3)
    axes[0].plot(bounds.s[index], index, 'k.')
axes[0].axvline(QUERY_S, color = 'C0', linestyle = '--', label = 'Query plane')
axes[0].set_xlabel('Survey s [m]')
axes[0].set_ylabel('Installed profile station')
axes[0].set_yticks(range(len(bounds)))
axes[0].set_title('Orange footprints intersect the query')
axes[0].legend()

########################################
# Inspect the returned polygon in that plane's local coordinates
########################################
axes[1].plot(polygon[:, 0] * 1e3, polygon[:, 1] * 1e3)
axes[1].set_xlabel('Local survey x [mm]')
axes[1].set_ylabel('Local survey y [mm]')
axes[1].set_title('Native combined cross-section')
axes[1].set_aspect('equal')
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
