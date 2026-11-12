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
# Reference and physical field
########################################
P0C             = 1.0e9                 # Proton reference p0*c [eV]
LENGTH          = 0.4                   # Local longitudinal interval [m]
MEAN_BY         = 0.02                  # Mean vertical field [T]
MEAN_GRADIENT   = 0.3                   # Mean dB_y/dx [T/m]
FIELD_ROLLS     = [0.0, np.pi / 4]      # Uniform dipole field rotations [rad]

########################################
# Numerical settings
########################################
N_STEPS         = 64                    # Boris steps across the element
N_FIELD_POINTS  = 101

################################################################################
# Specify a field in physical units
################################################################################

########################################
# Set boundary values, derivatives and mean
########################################
# Spline4 defines a quartic over one interval. mean is a field in tesla,
# not an integrated field. The on-axis profile and its derivative vanish
# at both ends.
by_profile = xt.Spline4(
    val_start   = 0.0,
    der_start   = 0.0,
    val_end     = 0.0,
    der_end     = 0.0,
    mean        = MEAN_BY,
)

########################################
# Build the Boris element
########################################
element = xt.SplineBoris(
    length  = LENGTH,
    by      = by_profile,
    n_steps = N_STEPS,
)

################################################################################
# Inspect the midplane field profile
################################################################################
# s_local runs from this element's entrance, not from the line's origin.
s_local     = np.linspace(0, LENGTH, N_FIELD_POINTS)
bx, by, bs  = element.get_field(x = 0.0, y = 0.0, s_local = s_local)
print('Entrance and exit B_y [T]:', by[0], by[-1])
print('Specified integral(B_y ds) [T m]:', MEAN_BY * LENGTH)
fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(s_local, by, label = 'SplineBoris B_y at x = y = 0')
ax.set_xlabel('s_local [m]')
ax.set_ylabel('B_y [T]')
ax.legend()
ax.grid()
fig.tight_layout()

################################################################################
# Track the on-axis, on-momentum proton
################################################################################

########################################
# Calculate the midplane momentum integral
########################################
reference   = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C)

# With y = 0, the normal dipole gives delta(P_x/p0) = -integral(B_y ds)/B_rho.
expected_px = -MEAN_BY * LENGTH / reference.rigidity0[0]

########################################
# Track and compare kinetic momentum
########################################
line        = xt.Line(elements = [element])
particles   = reference.copy()
line.track(particles)
print('Expected P_x/p0:', expected_px)
print('Tracked kinetic P_x/p0:', particles.kin_px[0])
print('Momentum difference:', particles.kin_px[0] - expected_px)
print('Exit x [m]:', particles.x[0])

################################################################################
# Add a transverse derivative and compute open optics
################################################################################

########################################
# Supply normal dipole and quadrupole profiles
########################################
# by[0] is B_y on axis [T]; by[1] is dB_y/dx on axis [T/m].
# Both profiles and their longitudinal derivatives vanish at the boundaries.
gradient_profile = xt.Spline4(
    val_start   = 0.0,
    der_start   = 0.0,
    val_end     = 0.0,
    der_end     = 0.0,
    mean        = MEAN_GRADIENT,
)
combined = xt.SplineBoris(
    length  = LENGTH,
    by      = (by_profile, gradient_profile),
    n_steps = N_STEPS,
)
x_scan = np.linspace(-1e-3, 1e-3, N_FIELD_POINTS)
bx_scan, by_scan, bs_scan = combined.get_field(
    x       = x_scan,
    y       = 0.0,
    s_local = LENGTH / 2,
)
fig_field, ax_field = plt.subplots(figsize = (8, 4))
ax_field.plot(x_scan * 1e3, by_scan)
ax_field.set_xlabel('x [mm]')
ax_field.set_ylabel('B_y at the longitudinal centre [T]')
ax_field.grid()
fig_field.tight_layout()

########################################
# Propagate entrance optics with the same reference
########################################
combined_line = xt.Line(
    elements        = [combined],
    particle_ref    = reference.copy(),
)
tw = combined_line.twiss(method = '4d', betx = 1.0, bety = 1.0)
tw.cols['name s x betx bety'].show()

################################################################################
# Cross-check a uniform rotated field against RBend
################################################################################

########################################
# Use common straight reference planes
########################################
# angle=0 fixes a straight reference, while k0 sets a nonzero physical field.
# Suppress bend edges: this is a homogeneous-field transport comparison.
comparison_errors = []
for roll in FIELD_ROLLS:
    constant_bx = -MEAN_BY * np.sin(roll)
    constant_by = MEAN_BY * np.cos(roll)
    uniform_field = xt.SplineBoris(
        length  = LENGTH,
        bx      = xt.Spline4(constant_bx, 0.0, constant_bx, 0.0, constant_bx),
        by      = xt.Spline4(constant_by, 0.0, constant_by, 0.0, constant_by),
        n_steps = N_STEPS,
    )
    bend = xt.RBend(
        length_straight     = LENGTH,
        angle               = 0.0,
        k0                  = MEAN_BY / reference.rigidity0[0],
        k0_from_h           = False,
        rot_s_rad           = roll,
        model               = 'bend-kick-bend',
        edge_entry_active   = False,
        edge_exit_active    = False,
    )
    launch = xt.Particles(
        mass0   = xt.PROTON_MASS_EV,
        p0c     = P0C,
        x       = 1e-3,
        px      = 1e-3,
        y       = 0.5e-3,
        py      = -0.5e-3,
        delta   = 0.01,
    )
    spline_particle = launch.copy()
    bend_particle = launch.copy()
    xt.Line(elements = [uniform_field]).track(spline_particle)
    xt.Line(elements = [bend]).track(bend_particle)
    differences = np.array([
        spline_particle.x[0] - bend_particle.x[0],
        spline_particle.y[0] - bend_particle.y[0],
        spline_particle.kin_px[0] - bend_particle.kin_px[0],
        spline_particle.kin_py[0] - bend_particle.kin_py[0],
    ])
    comparison_errors.append(differences)
    print('Field roll [rad]:', roll)
    print('Spline minus bend: x/y [m], kinetic px/py:', differences)
plt.show()
