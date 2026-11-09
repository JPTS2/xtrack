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
P0C     = 1.0e9                         # Proton reference p0*c [eV]
SPECIES = np.array(['proton', 'deuteron', 'He4'])
K0L     = 1.0e-4                        # Integrated dipole strength [rad]

################################################################################
# Define a common reference and a thin dipole
################################################################################
line = xt.Line(
    elements        = [xt.Multipole(knl = [K0L])],
    element_names   = ['dipole'],
)
line.set_particle_ref('proton', p0c = P0C)

################################################################################
# Compare species at the same velocity
################################################################################

########################################
# Build particles from their species names
########################################
# These are bare nuclei with charges +1e, +1e and +2e. build_particles
# resolves their tabulated masses, charge_ratio, mass_ratio and chi.
# delta = p*c/(mass_ratio*p0*c) - 1, so delta = 0 gives equal velocity.
particles_velocity  = line.build_particles(pdg_id = SPECIES, delta = 0.0)
pc_velocity         = (
    particles_velocity.mass_ratio * particles_velocity.p0c
    * (1 + particles_velocity.delta)
)

########################################
# Apply the same dipole kick
########################################
# With zero initial px and hxl = 0, the kick gives px = -chi*K0L.
# The normalized kick therefore depends on charge-to-mass ratio.
line.track(particles_velocity)
angle_velocity  = np.arctan(particles_velocity.kin_xp)
beta_velocity   = particles_velocity.beta0 * particles_velocity.rvv
velocity_table  = xt.Table(
    data    = {
        'species':      SPECIES,
        'mass_ratio':   particles_velocity.mass_ratio,
        'charge_ratio': particles_velocity.charge_ratio,
        'pc_GeV':       pc_velocity * 1e-9,
        'beta':         beta_velocity,
        'angle_mrad':   angle_velocity * 1e3,
    },
    index   = 'species',
)
print('Same velocity; different physical momenta:')
velocity_table.show(maxwidth = 'full')

################################################################################
# Compare species at the same physical momentum
################################################################################

########################################
# Adjust the mass-scaled momentum deviation
########################################
delta_momentum      = 1 / particles_velocity.mass_ratio - 1
particles_momentum  = line.build_particles(
    pdg_id  = SPECIES,
    delta   = delta_momentum,
)
pc_momentum = (
    particles_momentum.mass_ratio * particles_momentum.p0c
    * (1 + particles_momentum.delta)
)

########################################
# Compare the deflections
########################################
line.track(particles_momentum)
angle_momentum  = np.arctan(particles_momentum.kin_xp)
beta_momentum   = particles_momentum.beta0 * particles_momentum.rvv
momentum_table  = xt.Table(
    data    = {
        'species':    SPECIES,
        'delta':      particles_momentum.delta,
        'pc_GeV':     pc_momentum * 1e-9,
        'beta':       beta_momentum,
        'angle_mrad': angle_momentum * 1e3,
    },
    index   = 'species',
)
print('Same physical momentum; different velocities:')
momentum_table.show(maxwidth = 'full')

# At equal momentum, the equally charged proton and deuteron have equal
# deflections; the doubly charged helium nucleus has about twice their angle.
print('Angle ratios at equal momentum:', angle_momentum / angle_momentum[0])
