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
P0C                 = 1.0e9             # Proton momentum times c [eV]

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
reference   = xt.Particles('proton', p0c = P0C)
rigidity    = reference.rigidity0[0]

# At y = 0, B_y/(B*rho) = k0 + k1*x + k2*x^2/2 for this proton reference.
k0  = DIPOLE_FIELD / rigidity
k1  = QUAD_GRADIENT / rigidity
k2  = SEXT_DERIVATIVE / rigidity
print('Normalize physical fields to the reference rigidity'
      '\n---------------------------------------------------')
print('Positive-charge proton reference; strengths use its signed rigidity.')
print(f'Reference momentum [GeV/c]: {reference.p0c[0] * 1e-9:.3f}')
print(f'Reference rigidity [T m]: {rigidity:.6f}')
print(f'B_y [T]: {DIPOLE_FIELD:.3f} -> k0 [m^-1]: {k0:.6f}')
print(f'dB_y/dx [T/m]: {QUAD_GRADIENT:.3f} -> k1 [m^-2]: {k1:.6f}')
print(f'd^2 B_y/dx^2 [T/m^2]: {SEXT_DERIVATIVE:.3f} '
      f'-> k2 [m^-3]: {k2:.6f}')

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
print('\nDistributed magnets and integrated strengths'
      '\n--------------------------------------------')
print('Bend angle defines reference geometry; k0 specifies the powered field.')
print('Here the reference angle is chosen to equal k0 times the magnet length.')
print(f'Bend angle [rad]: {bend.angle:.6f}')
print(f'Integrated dipole k0 * L [1]: {k0 * bend.length:.6f}')
print(f'Integrated quadrupole k1 * L [m^-1]: {quad.k1 * quad.length:.6f}')
print(f'Integrated sextupole k2 * L [m^-2]: {sext.k2 * sext.length:.6f}')

########################################
# Integrated coefficients for a thin multipole
########################################
# knl[n] is integral(k_n ds), without dividing by n!. The factorial enters
# the field/kick polynomial. Here we combine quadrupole and sextupole terms.
thin = xt.Multipole(knl = [0.0, k1 * MAGNET_LENGTH, k2 * MAGNET_LENGTH])
print('\nThin multipole coefficients\n---------------------------')
print('This thin element combines quadrupole and sextupole terms, '
      'without a dipole.')
print('knl[n] stores the integral of k_n, without division by n!.')
print('The factorial belongs in the field and kick polynomial.')
print(f'knl[1] [m^-1]: {thin.knl[1]:.6f}')
print(f'knl[2] [m^-2]: {thin.knl[2]:.6f}')

################################################################################
# Inspect the horizontal thin kick at y = 0
################################################################################

########################################
# Calculate the expected normalized momentum change
########################################
# For the reference species, delta_px = -knl[1]*x - knl[2]*x^2/2.
quadrupole_px   = -thin.knl[1] * X_START
sextupole_px    = -thin.knl[2] * X_START**2 / 2
expected_px     = quadrupole_px + sextupole_px
print('\nPredict and track the thin kick\n-------------------------------')
print('px is canonical horizontal momentum divided by the reference momentum.')
print('At y = 0, the kick is -knl[1]*x - knl[2]*x**2/2.')
print(f'Initial x [m]: {X_START:.6f}')
print(f'Quadrupole contribution to px [1]: {quadrupole_px:.9e}')
print(f'Sextupole contribution to px [1]: {sextupole_px:.9e}')
print(f'Predicted px after the kick [1]: {expected_px:.9e}')

########################################
# Track through the thin element
########################################
line = xt.Line(elements = [thin])
line.set_particle_ref('proton', p0c = P0C)
particles = line.build_particles(x = X_START)
line.track(particles)
print(f'Tracked px after the kick [1]: {particles.px[0]:.9e}')
print(f'Tracked minus predicted px [1]: {particles.px[0] - expected_px:.3e}')
print(f'Unchanged x after the thin kick [m]: {particles.x[0]:.6f}')
