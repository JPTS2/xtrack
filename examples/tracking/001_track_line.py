# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Launch and propagation
########################################
N_TURNS         = 16
X               = np.array([0.1, 0.2, 0.3]) * 1e-3  # [m]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Build particles at the ring entrance
################################################################################
particles = line.build_particles(x = X, delta = 0.0)
print('Initial particles:')
particles.get_table().cols[
    'particle_id x px s at_turn at_element'].show(maxwidth = 'full')

################################################################################
# Track complete turns
################################################################################
# This magnetic ring has no RF or radiation. delta is conserved; zeta evolves.
# The tracker is built automatically on the default CPU context.
line.track(particles, num_turns = N_TURNS)

################################################################################
# Inspect the propagated particles
################################################################################
# Full turns return to element 0 and reset s to zero. at_turn counts the turns.
# These are final coordinates, not a history of the intervening motion.
particles.get_table().cols[
    'particle_id x px s at_turn at_element state'].show(maxwidth = 'full')
print('Final delta:', particles.delta)
