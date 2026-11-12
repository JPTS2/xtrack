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
# Common point cloud and reference particle
########################################
N_PARTICLES     = 1800
SEED            = 20261104
P0C             = 1e9                   # Proton reference momentum [eV/c]
X_SAMPLE_MAX    = 0.016                 # Uniform x half-range [m]
Y_SAMPLE_MAX    = 0.014                 # Uniform y half-range [m]

########################################
# Rectangle, ellipse, polygon and rounded-corner dimensions
########################################
HALF_X          = 0.012                 # Rectangle half-width / ellipse a [m]
HALF_Y          = 0.008                 # Rectangle half-height / ellipse b [m]
X_VERTICES      = [-0.012, 0.0, 0.012, 0.012, 0.0, -0.012]  # [m]
Y_VERTICES      = [-0.006, -0.010, -0.006, 0.006, 0.010, 0.006]  # [m]
CORNER_A        = 0.006                 # Rounded-corner ellipse a [m]
CORNER_B        = 0.005                 # Rounded-corner ellipse b [m]

########################################
# Shift and roll of a second rectangular cut
########################################
SHIFT_X         = 0.002                 # Aperture origin in entrance axes [m]
SHIFT_Y         = 0.001                 # [m]
ROLL            = np.deg2rad(25.0)      # [rad]; local x turns toward +y

################################################################################
# Define native tracking elements that cut transverse phase space
################################################################################

########################################
# Specify local transverse boundaries
########################################
rectangle   = xt.LimitRect(
    min_x   = -HALF_X,
    max_x   = HALF_X,
    min_y   = -HALF_Y,
    max_y   = HALF_Y,
)
ellipse     = xt.LimitEllipse(a = HALF_X, b = HALF_Y)
polygon     = xt.LimitPolygon(
    x_vertices  = X_VERTICES,
    y_vertices  = Y_VERTICES,
)
racetrack   = xt.LimitRacetrack(
    min_x   = -HALF_X,
    max_x   = HALF_X,
    min_y   = -HALF_Y,
    max_y   = HALF_Y,
    a       = CORNER_A,
    b       = CORNER_B,
)
shifted     = xt.LimitRect(
    min_x       = -HALF_X,
    max_x       = HALF_X,
    min_y       = -HALF_Y,
    max_y       = HALF_Y,
    shift_x     = SHIFT_X,
    shift_y     = SHIFT_Y,
    rot_s_rad   = ROLL,
)
apertures   = {
    'Rectangle': rectangle,
    'Ellipse': ellipse,
    'Polygon': polygon,
    'Racetrack': racetrack,
    'Shifted/rolled rectangle': shifted,
}
# These Limit elements act during tracking and set particle.state on loss.
# Chamber profiles and pipes are defined separately in 002.
# Ellipse a and b are semiaxes. Polygon vertices follow the boundary, with
# automatic closure from the last vertex to the first. Racetrack a/b round
# the four corners inside its min/max rectangle; they are not its outer size.
# The shifted cut interprets the same limits in its displaced, rotated axes.

########################################
# Generate one sample and give every cut an identical copy
########################################
rng         = np.random.default_rng(SEED)
x_initial   = rng.uniform(-X_SAMPLE_MAX, X_SAMPLE_MAX, N_PARTICLES)
y_initial   = rng.uniform(-Y_SAMPLE_MAX, Y_SAMPLE_MAX, N_PARTICLES)
initial     = xt.Particles(
    'proton',
    p0c = P0C,
    x   = x_initial,
    y   = y_initial,
)
alive_masks = {}
for name, aperture in apertures.items():
    particles   = initial.copy()
    aperture.track(particles)
    particles.sort(interleave_lost_particles = True)
    alive       = np.asarray(particles.state) > 0
    alive_masks[name]   = alive
    print(f'{name}: {np.count_nonzero(alive)}/{N_PARTICLES} points survive')
# Sorting restores particle-ID order, including losses. Counts describe this
# uniform geometric sample, not beam transmission or dynamic aperture.

################################################################################
# Compare survival regions using the same coordinates and colour scheme
################################################################################

########################################
# Draw the boundaries used by the native tracking elements
########################################
theta       = np.linspace(0, 2 * np.pi, 201)
outlines    = {
    'Rectangle': (
        [-HALF_X, HALF_X, HALF_X, -HALF_X, -HALF_X],
        [-HALF_Y, -HALF_Y, HALF_Y, HALF_Y, -HALF_Y]),
    'Ellipse': (HALF_X * np.cos(theta), HALF_Y * np.sin(theta)),
    'Polygon': (polygon.x_closed, polygon.y_closed),
}


# Join quarter-ellipses using the native racetrack's corner centres and radii.
corner_centres  = [
    (racetrack.max_x - racetrack.a, racetrack.max_y - racetrack.b),
    (racetrack.min_x + racetrack.a, racetrack.max_y - racetrack.b),
    (racetrack.min_x + racetrack.a, racetrack.min_y + racetrack.b),
    (racetrack.max_x - racetrack.a, racetrack.min_y + racetrack.b),
]
racetrack_x = []
racetrack_y = []
for index, (centre_x, centre_y) in enumerate(corner_centres):
    angle   = np.linspace(index * np.pi / 2, (index + 1) * np.pi / 2, 41)
    racetrack_x.extend(centre_x + racetrack.a * np.cos(angle))
    racetrack_y.extend(centre_y + racetrack.b * np.sin(angle))
outline_racetrack   = (racetrack_x + [racetrack_x[0]],
                     racetrack_y + [racetrack_y[0]])
outlines['Racetrack']   = outline_racetrack

# Transform the rectangular boundary FROM its aperture axes to the common
# entrance axes. Tracking applies the inverse transform to test each point.
rect_x, rect_y  = np.array(outlines['Rectangle'])
outlines['Shifted/rolled rectangle']    = (
    SHIFT_X + rect_x * np.cos(ROLL) - rect_y * np.sin(ROLL),
    SHIFT_Y + rect_x * np.sin(ROLL) + rect_y * np.cos(ROLL),
)

########################################
# Plot accepted and rejected input points in the common entrance axes
########################################
fig, axes   = plt.subplots(
    2, 3,
    sharex  = True,
    sharey  = True,
    figsize = (12, 8),
)
for axis, (name, alive) in zip(axes.flat, alive_masks.items()):
    axis.scatter(
        1e3 * x_initial[~alive], 1e3 * y_initial[~alive],
        s       = 5,
        color   = 'lightgrey',
        label   = 'Lost',
    )
    axis.scatter(
        1e3 * x_initial[alive], 1e3 * y_initial[alive],
        s       = 5,
        label   = 'Survives',
    )
    outline_x, outline_y    = outlines[name]
    axis.plot(
        1e3 * np.asarray(outline_x), 1e3 * np.asarray(outline_y),
        color   = 'black',
        label   = 'Boundary',
    )
    axis.set_title(f'{name}\n{np.count_nonzero(alive)} survive')
    axis.set_aspect('equal')
for axis in axes[:, 0]:
    axis.set_ylabel('Entrance y [mm]')
for axis in axes[-1, :-1]:
    axis.set_xlabel('Entrance x [mm]')
axes[-1, -1].axis('off')
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[-1, -1].legend(handles, labels, loc = 'center')
axes[-1, -1].text(
    0.5, 0.15,
    'All cuts use identical input points.\n'
    'Each boundary is drawn in entrance axes.',
    ha          = 'center',
    transform   = axes[-1, -1].transAxes,
)
fig.suptitle('Independent aperture cuts on identical input points')
fig.tight_layout(rect = (0, 0.03, 1, 0.96), h_pad = 3.0)
plt.show()
