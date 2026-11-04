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
P0C         = 1.0e9                     # Proton reference p0*c [eV]
PARTICLE_ID = np.array([42, 7, 19, 3])
FOLLOW_ID   = 42                        # Identifier to follow through sorting
X           = np.array([2e-3, -1e-3, 3e-3, 0.0])  # [m]
X_MIN       = 1.0e-3                    # Selection threshold [m]
X_SHIFT     = 1.0e-3                    # Shift only the selected copy [m]

################################################################################
# Create identified particles
################################################################################
particles = xt.Particles(
    pdg_id_0    = 'proton',
    p0c         = P0C,
    particle_id = PARTICLE_ID,
    x           = X,
)
particles.get_table().cols['particle_id x state'].show()
row_before  = np.flatnonzero(particles.particle_id == FOLLOW_ID)[0]
x_before    = particles.x[row_before]

################################################################################
# Sort and select
################################################################################

########################################
# Sort storage without changing particle identity
########################################
# Sorting is a CPU operation. Array position is not the particle identifier.
particles.sort(by = 'x')
print('Sorted by x:')
particles.get_table().cols['particle_id x state'].show()
row_after   = np.flatnonzero(particles.particle_id == FOLLOW_ID)[0]
x_after     = particles.x[row_after]
print(f'Particle {FOLLOW_ID}: storage row {row_before} -> {row_after}')
print(f'Its x coordinate is unchanged: {x_before:g} -> {x_after:g} m')

########################################
# Select an independent subset
########################################
selected = particles.filter(particles.x >= X_MIN)
selected.x += X_SHIFT
print('Shifted selected particles retain their identifiers:')
selected.get_table().cols['particle_id x state'].show()
print('The original coordinates are unchanged:')
particles.get_table().cols['particle_id x state'].show()

################################################################################
# Merge batches with colliding identifiers
################################################################################
# Combine two copies of the selected subset. Their ID ranges overlap, so merge
# shifts the later batch's valid IDs and corresponding parent IDs.
# Both copies have compatible scalar reference and clock fields, as required.
merged = xt.Particles.merge([selected, selected.copy()])
merged.get_table().cols['particle_id parent_particle_id x state'].show()
print('Merged particle count:', len(merged.particle_id))
print('Unique identifier count:', len(np.unique(merged.particle_id)))
