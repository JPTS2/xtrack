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
# Reusable transverse profiles
########################################
ROUND_RADIUS    = 0.025                 # Round-pipe radius [m]
ELLIPSE_A       = 0.025                 # Horizontal semiaxis [m]
ELLIPSE_B       = 0.012                 # Vertical semiaxis [m]

########################################
# Longitudinal stations in the pipe's own coordinates
########################################
TRANSITION_IN   = 1.0                   # End of round section [m]
TRANSITION_OUT  = 2.0                   # Start of elliptical section [m]
PIPE_LENGTH     = 3.0                   # [m]
PROFILE_POINTS  = 512                   # Points for displayed profile polygons

########################################
# Circular body and asymmetric winglets, in SVG millimetres
########################################
BODY_RADIUS_MM      = 25.0              # Radius of the circular body [mm]
OUTER_REACH_MM      = 55.0              # Outer tip at positive x [mm]
INNER_REACH_MM      = 40.0              # Inner tip at negative x [mm]
WING_HALF_HEIGHT_MM = 4.0               # Also the rounded-cap radius [mm]

########################################
# Convert and sample the SVG boundary
########################################
SVG_SCALE           = 1e-3              # Metres per SVG coordinate unit
SVG_RESOLUTION      = 2e-4              # Intermediate SVG sampling distance [m]

################################################################################
# Define reusable transverse profiles
################################################################################

########################################
# Store named shapes in the environment
########################################
env         = xt.Environment()
round_shape = env.aperture.new_profile(
    'round', 'Circle',
    radius  = ROUND_RADIUS,
)
flat_shape  = env.aperture.new_profile(
    'flat', 'Ellipse',
    a   = ELLIPSE_A,
    b   = ELLIPSE_B,
)
# Profiles describe transverse boundaries. They are not tracking elements.

########################################
# Define the winged profile with the same API, using an SVG path
########################################
join_x      = np.sqrt(BODY_RADIUS_MM**2 - WING_HALF_HEIGHT_MM**2)
outer_cap_x = OUTER_REACH_MM - WING_HALF_HEIGHT_MM
inner_cap_x = -INNER_REACH_MM + WING_HALF_HEIGHT_MM
# The circular body meets both winglets at y = +/- WING_HALF_HEIGHT_MM.
# For the shared ring's positive, untilted bends, +x is the outward side.

svg_path    = (
    f'M {join_x:.6f} {-WING_HALF_HEIGHT_MM} '
    f'L {outer_cap_x} {-WING_HALF_HEIGHT_MM} '
    f'A {WING_HALF_HEIGHT_MM} {WING_HALF_HEIGHT_MM} 0 0 1 '
    f'{outer_cap_x} {WING_HALF_HEIGHT_MM} '
    f'L {join_x:.6f} {WING_HALF_HEIGHT_MM} '
    f'A {BODY_RADIUS_MM} {BODY_RADIUS_MM} 0 0 1 '
    f'{-join_x:.6f} {WING_HALF_HEIGHT_MM} '
    f'L {inner_cap_x} {WING_HALF_HEIGHT_MM} '
    f'A {WING_HALF_HEIGHT_MM} {WING_HALF_HEIGHT_MM} 0 0 1 '
    f'{inner_cap_x} {-WING_HALF_HEIGHT_MM} '
    f'L {-join_x:.6f} {-WING_HALF_HEIGHT_MM} '
    f'A {BODY_RADIUS_MM} {BODY_RADIUS_MM} 0 0 1 '
    f'{join_x:.6f} {-WING_HALF_HEIGHT_MM} Z'
)
# M moves, L draws a line, A draws an elliptical arc, and Z closes the path.
# SVG y points down. SVGShape converts it to physical y pointing up; this
# vertically symmetric example therefore retains the same visible outline.

winged_shape    = env.aperture.new_profile(
    'winged', 'SVGShape',
    svg_path    = svg_path,
    scale       = SVG_SCALE,
    resolution  = SVG_RESOLUTION,
)
# SVGShape is another transverse profile type, alongside Circle and Ellipse.
# resolution samples the SVG first; PROFILE_POINTS resamples the final polygon.
print('SVG path (coordinates in mm):', svg_path)
print('SVG sampling distance [m]:', SVG_RESOLUTION)

################################################################################
# Compose pipes from profile positions
################################################################################

########################################
# Join round and elliptical sections through a transition
########################################
pipe    = env.aperture.new_pipe(
    'round_to_flat',
    positions   = [
        round_shape.place(at = 0.0),
        round_shape.place(at = TRANSITION_IN),
        flat_shape.place(at = TRANSITION_OUT),
        flat_shape.place(at = PIPE_LENGTH),
    ],
)

########################################
# Reuse the SVG profile at both ends of a uniform winged pipe
########################################
winged_pipe = env.aperture.new_pipe(
    'winged_tube',
    positions   = [
        winged_shape.place(at = 0),
        winged_shape.place(at = PIPE_LENGTH),
    ],
)
print('Profiles:', env.aperture.profiles.keys())
print('Pipes:', env.aperture.pipes.keys())
print('Round-to-flat pipe length [m]:', pipe.length)
for position in pipe:
    print('Station [m], profile:', position.shift_s, position.profile.name)
# A pipe combines profile positions and the connecting wall geometry. Here
# the section narrows vertically between the two transition stations.
# Installation on a line and creation of tracking limits are separate steps.

################################################################################
# Plot the native profiles and longitudinal pipe projections
################################################################################

########################################
# Compare the profile definitions
########################################
fig, axes   = plt.subplots(1, 3, figsize = (15, 5))
for ax, profile in zip(axes, [round_shape, flat_shape, winged_shape]):
    profile.plot(len_points = PROFILE_POINTS, ax = ax)
    ax.set_xlabel('Local x [m]')
    ax.set_ylabel('Local y [m]')
    ax.set_aspect('equal')
    ax.grid()
axes[2].annotate(
    'Outer winglet (+x)',
    xy          = (OUTER_REACH_MM * SVG_SCALE, 0),
    xytext      = (0.2 * OUTER_REACH_MM * SVG_SCALE,
                   1.4 * BODY_RADIUS_MM * SVG_SCALE),
    arrowprops  = {'arrowstyle': '->'},
)
axes[2].annotate(
    'Inner winglet (-x)',
    xy          = (-INNER_REACH_MM * SVG_SCALE, 0),
    xytext      = (-1.1 * INNER_REACH_MM * SVG_SCALE,
                   -1.5 * BODY_RADIUS_MM * SVG_SCALE),
    arrowprops  = {'arrowstyle': '->'},
)
axes[2].set_xlim(
    -1.4 * INNER_REACH_MM * SVG_SCALE, 1.2 * OUTER_REACH_MM * SVG_SCALE)
axes[2].set_ylim(
    -1.9 * BODY_RADIUS_MM * SVG_SCALE, 1.9 * BODY_RADIUS_MM * SVG_SCALE)
fig.tight_layout()

########################################
# Resolve the horizontal and vertical wall transitions
########################################
pipe_fig, pipe_axes = plt.subplots(1, 2, figsize = (10, 4))
for ax, plane in zip(pipe_axes, ['sx', 'sy']):
    pipe.plot(plane = plane, ax = ax)
    ax.set_aspect('auto')
    ax.set_xlabel('Pipe s [m]')
    ax.set_ylabel('Local ' + plane[1] + ' [m]')
    ax.set_title('Longitudinal ' + plane[1] + ' projection')
    ax.grid()
# The longitudinal panels use unequal axis scales to resolve the wall shape.
pipe_fig.tight_layout()
plt.show()
