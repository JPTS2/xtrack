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
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Launch and recording
########################################
X               = np.array([0.1, 3.0, 0.2]) * 1e-3  # [m]
N_TURNS         = 4
HALF_APERTURE   = 2.0e-3                # Horizontal half-aperture [m]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Create a loss to illustrate incomplete records
################################################################################
# A simple cut supplies a known loss; this is not an aperture-design or loss-
# refinement recipe. The lesson below is how to interpret the surviving record.
env.new(
    'aperture', xt.LimitRect,
    min_x   = -HALF_APERTURE,
    max_x   = HALF_APERTURE,
)
line.insert('aperture', at = 'bpm.s1.f@start')

################################################################################
# Track particles and record the ring entrance
################################################################################
particles = line.build_particles(x = X, delta = 0.0)
line.track(
    particles,
    num_turns               = N_TURNS,
    turn_by_turn_monitor    = True,
)
record = line.record_last_track

################################################################################
# Interpret missing samples and particle IDs
################################################################################

########################################
# Inspect the final particle container
########################################
# Live particles can be moved to the front of the container. Follow particle_id,
# not the array row. Lost particles retain their loss turn and element index.
particles.get_table().cols[
    'particle_id x at_turn at_element state'].show(maxwidth = 'full')
lost = particles.state <= 0
print('Lost particle IDs:', particles.particle_id[lost])
print('Loss element names:',
      [line.element_names[ii] for ii in particles.at_element[lost]])

########################################
# Inspect the record by particle-ID slot
########################################
# The particle lost near the ring entrance was sampled before the cut on turn 0.
# Later entrance slots for that ID are unfilled: state == 0, not a measurement
# at x == 0. The other two particle IDs still have all N_TURNS entrance samples.
print('Record state (particle-ID slot, turn):')
print(record.state)
for slot in range(len(X)):
    valid = record.state[slot] > 0
    print('ID, valid turns, x [m]:', slot,
          record.at_turn[slot, valid], record.x[slot, valid])
