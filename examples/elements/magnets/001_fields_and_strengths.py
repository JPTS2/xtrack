# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Reference particle
########################################
P0C                 = 1.0e9             # Proton reference p0*c [eV]

########################################
# Physical fields and magnetic lengths
########################################
DIPOLE_FIELD        = 0.2               # Vertical dipole field B_y [T]
QUAD_GRADIENT       = 1.0               # dB_y/dx [T/m]
SEXT_DERIVATIVE     = 3.0               # d^2 B_y/dx^2 [T/m^2]
MAGNET_LENGTH       = 0.5               # [m]

########################################
# Thin-kick probe
########################################
X_START             = 1.0e-3            # [m]; y = px = py = delta = 0

########################################
# Elliptic nonlinear lens
########################################
LENS_STRENGTH       = 1.0e-6            # Integrated knll [m]
LENS_SCALE          = 1.0e-2            # Singularities at x = +/-cnll [m]
LENS_AMPLITUDES     = np.array([0.1, 0.3, 0.6])  # x/cnll; y = 0

################################################################################
# Normalize fields to the reference rigidity
################################################################################
reference   = xt.Particles(mass0 = xt.PROTON_MASS_EV, p0c = P0C)
rigidity    = reference.rigidity0[0]

# At y = 0, B_y/(B*rho) = k0 + k1*x + k2*x^2/2 for this proton reference.
k0  = DIPOLE_FIELD / rigidity
k1  = QUAD_GRADIENT / rigidity
k2  = SEXT_DERIVATIVE / rigidity
print('Reference rigidity B*rho [T m]:', rigidity)
print('Normalized dipole k0 [m^-1]:', k0)
print('Normalized quadrupole k1 [m^-2]:', k1)
print('Normalized sextupole k2 [m^-3]:', k2)

################################################################################
# Define the corresponding magnetic elements
################################################################################

########################################
# Distributed body strengths
########################################
# In the bend, angle sets reference geometry; k0 specifies the field.
bend = xt.Bend(
    length  = MAGNET_LENGTH,
    angle   = k0 * MAGNET_LENGTH,
    k0      = k0,
)
quad = xt.Quadrupole(length = MAGNET_LENGTH, k1  = k1)
sext = xt.Sextupole(length = MAGNET_LENGTH, k2  = k2)
print('Bend angle [rad] and integral(k0 ds) [1]:',
      bend.angle, k0 * bend.length)
print('Quadrupole integrated strength [m^-1]:', quad.k1 * quad.length)
print('Sextupole integrated strength [m^-2]:', sext.k2 * sext.length)

########################################
# Integrated coefficients for a thin multipole
########################################
# knl[n] is integral(k_n ds), without dividing by n!. The factorial enters
# the field/kick polynomial. Here we combine quadrupole and sextupole terms.
thin = xt.Multipole(knl = [0.0, k1 * MAGNET_LENGTH, k2 * MAGNET_LENGTH])
print('Thin knl[1] [m^-1]:', thin.knl[1])
print('Thin knl[2] [m^-2]:', thin.knl[2])

################################################################################
# Inspect the horizontal thin kick at y = 0
################################################################################

########################################
# Calculate the expected normalized momentum change
########################################
# For the reference species, delta_px = -knl[1]*x - knl[2]*x^2/2.
expected_px = -thin.knl[1] * X_START - thin.knl[2] * X_START**2 / 2

########################################
# Track through the thin element
########################################
line = xt.Line(elements = [thin])
line.set_particle_ref('proton', p0c = P0C)
particles = line.build_particles(x = X_START)
line.track(particles)
print('Expected horizontal kick P_x/p0:', expected_px)
print('Tracked horizontal momentum P_x/p0:', particles.px[0])
print('Horizontal position before and after [m]:', X_START, particles.x[0])

################################################################################
# Track an elliptic nonlinear lens
################################################################################

########################################
# Launch inside the two singular points
########################################
# This potential is non-polynomial. Keep |x/cnll| < 1 at y = 0; its small-
# amplitude quadrupole term has integrated strength 2*knll/cnll**2 [m^-1].
lens = xt.NonLinearLens(knll = LENS_STRENGTH, cnll = LENS_SCALE)
lens_line = xt.Line(elements = [lens])
lens_line.set_particle_ref('proton', p0c = P0C)
lens_particles = lens_line.build_particles(x = LENS_SCALE * LENS_AMPLITUDES)
lens_line.track(lens_particles)

########################################
# Compare with the exact horizontal-axis derivative
########################################
# For u = x/cnll, the potential on this interval is
# U(u) = -u*asin(u)/sqrt(1-u**2); the kick is knll/cnll * dU/du.
u = LENS_AMPLITUDES
lens_expected_px = -LENS_STRENGTH / LENS_SCALE * (
    np.arcsin(u) / (1 - u**2)**1.5 + u / (1 - u**2))
linear_px = -2 * LENS_STRENGTH / LENS_SCALE * u
print('Elliptic lens x/cnll:', u)
print('Exact elliptic kick P_x/p0:', lens_expected_px)
print('Tracked elliptic kick P_x/p0:', lens_particles.px)
print('Small-amplitude quadrupole approximation P_x/p0:', linear_px)
