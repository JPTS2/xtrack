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
# Launch and tracking
########################################
N_TURNS         = 4
X               = 1.0e-4                # [m]
PX              = 1.0e-6                # Normalized canonical momentum
Y               = 2.0e-4                # [m]
PY              = 3.0e-6                # Normalized canonical momentum
ZETA            = 1.0e-2                # [m]
DELTA           = 1.0e-4
STOP_ELEMENT    = 'obs.4'

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Build particles and save the initial coordinates
################################################################################
particles = line.build_particles(
    x       = X,
    px      = PX,
    y       = Y,
    py      = PY,
    zeta    = ZETA,
    delta   = DELTA,
)
initial     = particles.copy()
coordinates = np.array(['x', 'px', 'y', 'py', 'zeta', 'delta'])

################################################################################
# Reverse complete turns
################################################################################

########################################
# Track forward
########################################
# This static magnetic ring has no radiation, stochastic kicks or losses.
# Backtracking applies inverse element maps; it is not a model of a beam whose
# velocity has been reversed in the same magnetic field.
line.track(particles, num_turns = N_TURNS)
particles.get_table().cols[
    'x px y py zeta delta at_turn'].show(maxwidth = 'full')

########################################
# Track backward and inspect the residuals
########################################
line.track(particles, num_turns = N_TURNS, backtrack = True)
residuals = xt.Table(
    data    = {
        'coordinate': coordinates,
        'difference': np.array([
            float(getattr(particles, nn)[0] - getattr(initial, nn)[0])
            for nn in coordinates
        ]),
    },
    index   = 'coordinate',
)

# Differences have units m for x, y, zeta; px, py and delta are dimensionless.
residuals.show(maxwidth = 'full')
print('Restored turn and element:', particles.at_turn, particles.at_element)

################################################################################
# Reverse a partial path
################################################################################

########################################
# Track to the second observation point
########################################
partial = initial.copy()
line.track(partial, ele_stop = STOP_ELEMENT)
partial.get_table().cols['x px s at_element'].show(maxwidth = 'full')

########################################
# Invert the same interval
########################################
# Use the same forward interval: backtrack starts at ele_stop and ends at 0.
line.track(partial, ele_stop = STOP_ELEMENT, backtrack = True)
partial.get_table().cols['x px s at_element'].show(maxwidth = 'full')
print('Initial x, px [m, 1]:', initial.x, initial.px)
