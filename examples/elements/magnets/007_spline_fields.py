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
plt.show()
