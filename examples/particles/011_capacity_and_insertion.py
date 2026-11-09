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
# Reference and reserved storage
########################################
P0C         = 1e9                       # Electron reference p0*c [eV]
CAPACITY    = 8                         # Live, lost and unused array slots
X_INITIAL   = np.array([-0.3, -0.1, 0.1, 0.3]) * 1e-3  # [m]

########################################
# Additional prepared particles
########################################
X_ADDED     = np.array([0.4, 0.5]) * 1e-3  # [m]
INPUT_IDS   = np.array([100, 101])       # IDs in the incoming container
LOST_ID     = 1                         # Keep this original history as lost

################################################################################
# Reserve storage and retain a lost-particle history
################################################################################

########################################
# Populate four of the allocated slots
########################################
particles   = xt.Particles(
    _capacity   = CAPACITY,
    pdg_id_0    = 'electron',
    p0c         = P0C,
    x           = X_INITIAL,
)
print('Initial occupied slots:')
particles.remove_unused_space().get_table().cols[
    'particle_id x state'].show(maxwidth = 'full')

########################################
# Mark one history lost without deleting its occupied slot
########################################
particles.state[particles.particle_id == LOST_ID] = 0
# A lost history still consumes capacity; it is not an unused insertion slot.

################################################################################
# Add particles with the same reference and inspect native ID assignment
################################################################################

########################################
# Prepare a separate incoming container
########################################
incoming    = xt.Particles(
    pdg_id_0    = 'electron',
    p0c         = P0C,
    x           = X_ADDED,
    particle_id = INPUT_IDS,
)
print('Incoming container IDs:', incoming.particle_id)

# This is container insertion, not particle production or a physical collision.
# Coordinates are copied in the receiving reference frame, without transport.

########################################
# Let add_particles append and reorganize the occupied histories
########################################
particles.add_particles(incoming)
occupied    = particles.remove_unused_space()
print('After insertion; x [m]:')
occupied.get_table().cols['particle_id x state'].show(maxwidth = 'full')

# New IDs follow the largest existing ID; incoming IDs are not retained.
# Array order can change as live and lost histories are reorganized.

################################################################################
# Count live, lost and unused slots separately
################################################################################
print('Allocated / occupied / unused slots:', len(particles.x),
      len(occupied.x), len(particles.x) - len(occupied.x))
print('Live / lost histories:', np.count_nonzero(occupied.state > 0),
      np.count_nonzero(occupied.state <= 0))

# remove_unused_space returns an independent compact container for inspection;
# the original remains allocated at CAPACITY. Insufficient unused space causes
# add_particles to raise an error instead of growing that allocation.
