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
# Rectangle, ellipse and polygon dimensions
########################################
HALF_X          = 0.012                 # Rectangle half-width / ellipse a [m]
HALF_Y          = 0.008                 # Rectangle half-height / ellipse b [m]
X_VERTICES      = [-0.012, 0.0, 0.012, 0.012, 0.0, -0.012]  # [m]
Y_VERTICES      = [-0.006, -0.010, -0.006, 0.006, 0.010, 0.006]  # [m]

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
apertures   = {'Rectangle': rectangle, 'Ellipse': ellipse, 'Polygon': polygon}
# These Limit elements act during tracking and set particle.state on loss.
# Chamber profiles and pipes are defined separately in 002.
# Ellipse a and b are semiaxes. Polygon vertices follow the boundary, with
# automatic closure from the last vertex to the first.

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

########################################
# Plot accepted and rejected points in each local transverse plane
########################################
fig, axes   = plt.subplots(
    1, 3,
    sharex  = True,
    sharey  = True,
    figsize = (12, 4),
)
for axis, (name, alive) in zip(axes, alive_masks.items()):
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
    axis.set_title(f'{name}: {np.count_nonzero(alive)} survive')
    axis.set_xlabel('Local x [mm]')
    axis.set_aspect('equal')
axes[0].set_ylabel('Local y [mm]')
axes[-1].legend(loc = 'lower right')
fig.suptitle('Independent aperture cuts on identical input points')
fig.tight_layout(rect = (0, 0.03, 1, 0.96))
plt.show()
