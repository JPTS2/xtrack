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
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Execution
########################################
N_THREADS       = 0                     # 0: serial CPU; >0: OpenMP threads
X               = 1.0e-4                # [m]

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Select a CPU context
################################################################################
context = xo.ContextCpu(omp_num_threads = N_THREADS)
line.build_tracker(_context = context)
particles = line.build_particles(x = X)
line.track(particles)
print('Turns before editing:', particles.at_turn)

################################################################################
# Edit the line and rebuild through optics
################################################################################

########################################
# Invalidate the existing tracker
########################################
line.append('end_marker', xt.Marker())
print('Tracker invalidated:', line.tracker is None)

########################################
# Rebuild on the selected context
########################################
tw = line.twiss4d()
# Context identity is inspected here to demonstrate reuse after the rebuild.
print('Tracker retains the selected context:', line._context is context)
print('Tunes after adding a marker:', tw.qx, tw.qy)

################################################################################
# Continue with the original particle container
################################################################################
line.track(particles)
particles.get_table().cols['particle_id x px at_turn'].show()
