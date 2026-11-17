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
SPECIES = 'proton'
P0C     = 1.0e9                         # Reference p0*c [eV]

################################################################################
# Define the reference
################################################################################

########################################
# Specify momentum and species
########################################
reference = xt.Particles(pdg_id_0 = SPECIES, p0c = P0C)

########################################
# Inspect reference quantities
########################################
print('Define the reference\n--------------------')
print(f'Reference charge [e]: {reference.q0:g}')
print(f'Rest energy m0*c^2 [GeV]: {reference.mass0 * 1e-9:.9f}')
print(f'Reference momentum [GeV/c]: {reference.p0c[0] * 1e-9:.9f}')
print(f'Total energy [GeV]: {reference.energy0[0] * 1e-9:.9f}')
print(f'Kinetic energy [GeV]: {reference.kinetic_energy0[0] * 1e-9:.9f}')
print(f'beta0: {reference.beta0[0]:.9f}')
print(f'gamma0: {reference.gamma0[0]:.9f}')
print(f'Signed reference rigidity [T m]: {reference.rigidity0[0]:.9f}')

################################################################################
# Specify the same reference through kinetic energy
################################################################################

########################################
# Construct an equivalent reference
########################################
reference_from_energy = xt.Particles(
    pdg_id_0        = SPECIES,
    kinetic_energy0 = reference.kinetic_energy0[0],
)

########################################
# Compare the momenta
########################################
print('\nSpecify the same reference through kinetic energy\n'
      '-------------------------------------------------')
print(f'From momentum [GeV/c]: {reference.p0c[0] * 1e-9:.9f}')
print(f'From kinetic energy [GeV/c]: '
      f'{reference_from_energy.p0c[0] * 1e-9:.9f}')
