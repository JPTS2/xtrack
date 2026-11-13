# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Existing tracking limits in a straight line
########################################
P0C             = 1e9                   # Electron reference p0*c [eV]
STATION_START   = 1.0                   # Two co-located limits [m]
STATION_END     = 3.0                   # Final ellipse [m]
LINE_LENGTH     = 4.0                   # [m]
RECT_HALF_X     = 0.030                 # [m]
RECT_HALF_Y     = 0.020                 # [m]
ELLIPSE_A       = 0.020                 # Entrance ellipse semiaxis [m]
ELLIPSE_B       = 0.010                 # Entrance ellipse semiaxis [m]
EXIT_A          = 0.015                 # Exit ellipse semiaxis [m]
EXIT_B          = 0.008                 # Exit ellipse semiaxis [m]

########################################
# Magnet-associated input geometry
########################################
MAGNET_LENGTH   = 0.4                   # [m]

########################################
# Geometry sampling
########################################
TRANSITION_TOL  = 1e-3                  # Samples either side of a bound [m]
SAMPLE_STEP     = 0.25                  # Additional regular sampling [m]

################################################################################
# Start from discrete tracking elements, without a chamber model
################################################################################
line    = xt.Line(elements = {
    'drift.in': xt.Drift(length = STATION_START),
    'wide': xt.LimitRect(
        min_x   = -RECT_HALF_X,
        max_x   = RECT_HALF_X,
        min_y   = -RECT_HALF_Y,
        max_y   = RECT_HALF_Y,
    ),
    'narrow': xt.LimitEllipse(a = ELLIPSE_A, b = ELLIPSE_B),
    'drift.middle': xt.Drift(length = STATION_END - STATION_START),
    'exit': xt.LimitEllipse(a = EXIT_A, b = EXIT_B),
    'drift.out': xt.Drift(length = LINE_LENGTH - STATION_END),
})
line.set_particle_ref('electron', p0c = P0C)

################################################################################
# Inspect axis cuts at elements and their interpolated table values
################################################################################
context     = xo.ContextCpu()
context.allow_kernel_compilation    = True
line.build_tracker(_context = context)
# This diagnostic uses frozen-coordinate tracking and needs its own kernel.
table       = line.get_aperture_table()
table.cols['name s x_aper_low x_aper_high '
           'x_aper_low_discrete x_aper_high_discrete'].show(maxwidth = 'full')
print('Vertical discrete cuts at the co-located limits [m]:')
table.rows[['wide', 'narrow']].cols[
    'name s y_aper_low_discrete y_aper_high_discrete'].show()
# Discrete columns are NaN at elements without a cut. The other columns
# interpolate neighbouring axis cuts and extend endpoint values. At equal s,
# each actual limit keeps its own discrete values; it is not collapsed away.
# This diagnostic samples x with y=0, and y with x=0, on a 1mm grid. Its
# half-grid boundary estimates are not exact geometry or beam acceptance.

################################################################################
# Reconstruct chamber profiles on a separate copy of the line
################################################################################

########################################
# Import the installed limits through the native aperture container
########################################
reconstructed   = line.copy()
reconstructed.aperture.from_limits(with_progress = False)
reconstructed.aperture.get_bounds_table().cols[
    'name pipe_name profile_name s'].show(maxwidth = 'full')
# The source line still has only its original tracking limits. Reconstruction
# supplies profiles at those stations, not knowledge of the intervening wall.

########################################
# Sample around bounds as well as on a regular longitudinal grid
########################################
samples     = reconstructed.aperture.get_s_around_transitions(
    tol         = TRANSITION_TOL,
    resolution  = SAMPLE_STEP,
    s_range     = (STATION_START, STATION_END),
)
sections    = reconstructed.aperture.get_cross_sections_at_s(
    samples, extents = True)
sections.cols['s min_x max_x min_y max_y'].show(maxwidth = 'full')
# Between these stations the native model interpolates the limiting ellipse
# profiles. No chamber is specified upstream of the first installed profile.

################################################################################
# Import aperture definitions associated with magnets
################################################################################

########################################
# Store a separate shape definition and associate it with each magnet
########################################
associated_env = xt.Environment()
associated_env.new('quad.aperture', xt.LimitEllipse,
                   a = ELLIPSE_A, b = ELLIPSE_B)
for name in ['quad.one', 'quad.two']:
    associated_env.new(name, xt.Quadrupole, length = MAGNET_LENGTH)
    associated_env.elements[name].name_associated_aperture = 'quad.aperture'
associated_line = associated_env.new_line(
    length      = LINE_LENGTH,
    components  = [
        associated_env.place('quad.one', at = STATION_START, anchor = 'start'),
        associated_env.place('quad.two', at = STATION_END, anchor = 'start'),
    ],
)
# The shape is in the registry, not the tracking sequence. This metadata can
# also come from legacy data loaded without installing separate Limit elements.
print('Associated-data line:')
associated_line.get_table().cols['name s element_type'].show()

########################################
# Reconstruct each magnet interval rather than interpolate limit stations
########################################
associated_line.aperture.from_associated_apertures(with_progress = False)
associated_line.aperture.get_bounds_table().cols[
    'name pipe_name profile_name s'].show(maxwidth = 'full')
# Each pipe spans its associated magnet. This input says nothing about
# the connecting drifts; no wall or tracking loss is inferred for those gaps.
print('Associated sections at magnet centres:')
associated_line.aperture.get_cross_sections_at_s(
    [STATION_START + MAGNET_LENGTH / 2,
     STATION_END + MAGNET_LENGTH / 2],
    extents = True,
).cols['s min_x max_x min_y max_y'].show()
# Constructing the chamber does not install particle-removing tracking limits.

################################################################################
# Compare sampled axis-cut estimates with reconstructed chamber sections
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (11, 4))
axes[0].plot(table.s, table.x_aper_high * 1e3, 'o--',
             label = 'Interpolated axis-cut table')
axes[0].plot(table.s, table.x_aper_high_discrete * 1e3, 'kx',
             label = 'Discrete tracking cuts')
axes[0].plot(sections.s, sections.max_x * 1e3, '-',
             label = 'Reconstructed chamber')
axes[0].set_xlabel('s [m]')
axes[0].set_ylabel('Positive horizontal extent [mm]')
axes[0].legend()
for index in [0, len(samples) // 2, len(samples) - 1]:
    points  = sections.cross_section[index]
    closed  = np.vstack([points, points[0]])
    axes[1].plot(closed[:, 0] * 1e3, closed[:, 1] * 1e3,
                 label = f's = {samples[index]:g} m')
axes[1].set_xlabel('Local x [mm]')
axes[1].set_ylabel('Local y [mm]')
axes[1].set_aspect('equal')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
