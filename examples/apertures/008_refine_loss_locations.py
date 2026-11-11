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
# Straight geometry and entrance particles
########################################
LENGTH          = 2.0                   # [m]
HALF_X_START    = 0.02                  # [m]
HALF_X_END      = 0.01                  # [m]
HALF_Y          = 0.05                  # [m]
P0C             = 1e9                   # Proton reference momentum [eV/c]
# Use nonnegative px with |px| < 1 for this positive-wall comparison.
PX_START        = np.array([0, .003, .006, .008, .01, .015])  # p_x/p0

########################################
# Refinement resolution
########################################
N_THETA         = 72                    # Aperture polygon angular samples
R_MAX           = 0.10                  # Beyond aperture radius [m]
DR              = 1.0e-6                # Radial aperture sampling [m]
DS              = 0.02                  # Longitudinal aperture spacing [m]

################################################################################
# Track with aperture checks only at the two installed stations
################################################################################

########################################
# Build the two-station line
########################################
line    = xt.Line(
    elements        = [
        xt.LimitRect(
            min_x   = -HALF_X_START,
            max_x   = HALF_X_START,
            min_y   = -HALF_Y,
            max_y   = HALF_Y,
        ),
        xt.Drift(length = LENGTH, model = 'exact'),
        xt.LimitRect(
            min_x   = -HALF_X_END,
            max_x   = HALF_X_END,
            min_y   = -HALF_Y,
            max_y   = HALF_Y,
        ),
    ],
    element_names   = ['aper.start', 'drift', 'aper.end'],
)
line.set_particle_ref('proton', p0c = P0C)

########################################
# Launch and coarse-track the common particles
########################################
# x=y=py=delta=0 initially; px is normalized canonical momentum, not x'.
# The exact drift gives x(s)=s*px/sqrt(1-px^2). No RF or radiation.
initial = line.build_particles(
    x       = 0.0,
    y       = 0.0,
    px      = PX_START,
    py      = 0.0,
    delta   = 0.0,
)
coarse  = initial.copy()
line.track(coarse)

################################################################################
# Refine losses under an explicit interpolated-wall assumption
################################################################################
# Tracking checks the installed apertures, so every loss initially has s=LENGTH.
# Refinement estimates where a particle first intersects interpolated apertures
# between those stations. Here the horizontal wall at y=0 tapers linearly.
# This is a geometric assumption, not a measurement of an unknown chamber wall.
refined     = coarse.copy()
refinement  = xt.LossLocationRefinement(
    line,
    n_theta = N_THETA,
    r_max   = R_MAX,
    dr      = DR,
    ds      = DS,
)
refinement.refine_loss_location(refined)

################################################################################
# Compare with the analytical intersection, preserving particle identity
################################################################################

########################################
# Derive the wall intersections for these straight rays
########################################
slope               = PX_START / np.sqrt(1 - PX_START**2)
wall_gradient       = (HALF_X_START - HALF_X_END) / LENGTH
crossing_s          = HALF_X_START / (slope + wall_gradient)
# A crossing beyond the downstream station means survival over this line.
analytical_loss_s   = np.where(crossing_s < LENGTH, crossing_s, np.nan)

########################################
# Compare recorded losses in particle-ID order
########################################
coarse.sort(interleave_lost_particles = True)
refined.sort(interleave_lost_particles = True)
coarse_s    = np.where(coarse.state == 0, coarse.s, np.nan)
refined_s   = np.where(refined.state == 0, refined.s, np.nan)
loss_table  = xt.Table(
    data    = {
        'particle_id':  refined.particle_id,
        'state':        refined.state,
        'coarse_s':     coarse_s,
        'refined_s':    refined_s,
        'analytic_s':   analytical_loss_s,
        'error_s':      refined_s - analytical_loss_s},
    index   = 'particle_id',
)
print('Loss distances and errors [m]; NaN means survival over this line.')
loss_table.show(maxwidth = 'full')
print('Longitudinal sampling DS [m]:', DS)
# This is a CPU refinement of discrete Limit elements under the interpolated
# wall assumption. It does not recover an unknown chamber or model scattering.
# Polygon/radial and longitudinal sampling set the numerical resolution.

################################################################################
# Plot the assumed walls and the coarse and refined loss positions
################################################################################
fig, ax = plt.subplots(figsize = (9, 5))
s_plot  = np.linspace(0, LENGTH, 100)
wall    = HALF_X_START - wall_gradient * s_plot
ax.plot(s_plot, 1e3 * wall, 'k-', label = 'Assumed chamber wall')
ax.plot(s_plot, -1e3 * wall, 'k-')
for particle_id in range(len(PX_START)):
    stop    = min(crossing_s[particle_id], LENGTH)
    ax.plot(
        [0, stop], [0, 1e3 * slope[particle_id] * stop],
        color   = '0.7',
    )
lost    = coarse.state == 0
ax.plot(coarse.s[lost], 1e3 * coarse.x[lost], 'rx', label = 'Coarse loss')
lost    = refined.state == 0
ax.plot(refined.s[lost], 1e3 * refined.x[lost], 'o', label = 'Refined loss')
ax.set_xlabel('s [m]')
ax.set_ylabel('x [mm]')
ax.grid()
ax.legend()
fig.tight_layout()
plt.show()
