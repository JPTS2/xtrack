# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
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
