# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Reference and launch
########################################
P0C             = 1.0e9                 # Electron momentum [eV/c]
X               = [-2.0e-4, 2.0e-3, 2.0e-4]  # [m]
PX              = 1.0e-4                # Horizontal momentum Px / p0
HALF_APERTURE   = 1.0e-3                # Horizontal half-aperture [m]
DRIFT_LENGTH    = 1.0                   # [m]

########################################
# Destination execution
########################################
N_THREADS       = 2                     # OpenMP CPU threads

################################################################################
# Select the source and destination contexts
################################################################################
source_context      = xo.ContextCpu(omp_num_threads = 0)
destination_context = xo.ContextCpu(omp_num_threads = N_THREADS)
# CUDA alternative: destination_context = xo.ContextCupy()
# OpenCL alternative: destination_context = xo.ContextPyopencl()
# Both default contexts use host memory. A GPU context also requires data transfers.

################################################################################
# Convert between NumPy and context arrays
################################################################################

########################################
# Create an independent context array
########################################
host_x      = np.array(X)
context_x   = source_context.nparray_to_context_array(host_x, copy = True)

########################################
# Take an independent NumPy snapshot
########################################
snapshot_x  = source_context.nparray_from_context_array(context_x, copy = True)
print('Launch x [m], copied back to NumPy:', snapshot_x)
# On CPU, copy=True prevents these arrays from sharing storage.

################################################################################
# Create a loss before transferring the particles
################################################################################
particles   = xt.Particles(
    'electron',
    _context    = source_context,
    p0c         = P0C,
    x           = context_x,
    px          = PX,
)
aperture    = xt.LimitRect(
    _context    = source_context,
    min_x       = -HALF_APERTURE,
    max_x       = HALF_APERTURE,
)
aperture.track(particles)
print('Source after the aperture:')
particles.get_table().cols['particle_id state x px s'].show()

################################################################################
# Copy to another context and track independently
################################################################################

########################################
# Allocate the copy and its tracking element on the destination
########################################
copied      = particles.copy(_context = destination_context)
drift       = xt.Drift(
    _context    = destination_context,
    length      = DRIFT_LENGTH,
)

########################################
# Advance only the copy
########################################
drift.track(copied)
print('Copy after destination tracking; lost particles do not advance:')
copied.get_table().cols['particle_id state x px s'].show()
print('The source coordinates are unchanged:')
particles.get_table().cols['particle_id state x px s'].show()
# Rows can be reordered during transfers. Match particles by particle_id.

################################################################################
# Move the original object to the destination context
################################################################################
original    = particles
particles.move(_context = destination_context)
print('Move keeps the same Python object:', particles is original)
drift.track(particles)
print('Moved original after the same destination tracking:')
particles.get_table().cols['particle_id state x px s'].show()

################################################################################
# Return to the source context and resume tracking there
################################################################################

########################################
# Copy back while retaining the destination copy
########################################
returned        = copied.copy(_context = source_context)
source_drift    = drift.copy(_context = source_context)
source_drift.track(returned)
print('Returned copy after one more source-context drift:')
returned.get_table().cols['particle_id state x px s'].show()

########################################
# Move the original back and apply the same drift
########################################
particles.move(_context = source_context)
source_drift.track(particles)
print('Returned original after the same source-context drift:')
particles.get_table().cols['particle_id state x px s'].show()
# With a GPU destination, these return paths transfer data back to CPU.
# Conversion, movement and tracking costs are not measured by this tiny sample.
