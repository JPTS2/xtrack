# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xobjects as xo
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
# CPU tracking
########################################
N_THREADS       = 0                     # 0: serial; e.g. 2: OpenMP threads
N_TURNS         = 2                     # Turns before and after the edit
X               = [1.0e-4, 2.0e-4]      # Initial horizontal positions [m]

################################################################################
# Load the model
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]

################################################################################
# Select a CPU context and track
################################################################################
# OpenMP distributes particles over threads. This tiny batch illustrates the
# setup; it is not a performance comparison.
context     = xo.ContextCpu(omp_num_threads = N_THREADS)
print('Selected context:', context)
line.build_tracker(_context = context)
particles   = line.build_particles(x = X)
line.track(particles, num_turns = N_TURNS)
print('Completed turns before the edit:', particles.at_turn)

################################################################################
# Edit the line and rebuild on the same context
################################################################################

########################################
# Add a marker
########################################
# Structural edits invalidate the tracker. A marker adds no length or kick.
line.append('end_marker', xt.Marker())
print('Tracker invalidated:', line.tracker is None)

########################################
# Build the tracker explicitly
########################################
# Keep the context object and pass it again when rebuilding.
line.build_tracker(_context = context)
print('Rebuilt tracker available:', line.tracker is not None)

################################################################################
# Use the rebuilt line
################################################################################

########################################
# Compute optics
########################################
# The demo ring has no RF focusing at its default operating point.
tw  = line.twiss4d()
print(f'Tunes after adding the marker: Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}')

########################################
# Continue tracking the original particles
########################################
line.track(particles, num_turns = N_TURNS)
particles.get_table().cols['particle_id x px at_turn'].show(maxwidth = 'full')
