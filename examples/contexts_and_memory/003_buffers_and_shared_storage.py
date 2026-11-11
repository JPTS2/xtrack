# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################
BUFFER_BYTES    = 1024 * 1024            # Initial buffer capacity [bytes]
P0C             = 1.0e9                 # Reference p0*c [eV]
QUAD_LENGTH     = 0.25                  # [m]
K1              = 0.8                   # [m^-2]
CHANGED_K1      = 1.2                   # [m^-2]
X_INITIAL       = 1.0e-3                # [m]
X_CHANGED       = 2.0e-3                # [m]

################################################################################
# Allocate several objects in one CPU buffer
################################################################################
context     = xo.ContextCpu()
buffer      = context.new_buffer(capacity = BUFFER_BYTES)
# These separately allocated objects have independent fields.
# Sharing a buffer alone does not make their field values aliases.
quad_a      = xt.Quadrupole(
    _buffer = buffer,
    length  = QUAD_LENGTH,
    k1      = K1,
)
quad_b      = xt.Quadrupole(
    _buffer = buffer,
    length  = QUAD_LENGTH,
    k1      = K1,
)
particles   = xt.Particles(
    'electron',
    _buffer     = buffer,
    p0c         = P0C,
    x           = [X_INITIAL],
)

################################################################################
# Sharing a buffer does not share field values
################################################################################

########################################
# Change one of the two elements
########################################
quad_a.k1   = CHANGED_K1
print('Elements in the same buffer:')
print(f'  Changed quad_a k1: {quad_a.k1:g} m^-2')
print(f'  Unchanged quad_b k1: {quad_b.k1:g} m^-2')

########################################
# Copy particles into the same buffer
########################################
particles_copy      = particles.copy(_buffer = buffer)
particles_copy.x[0] = X_CHANGED
print('Particle batches in the same buffer:')
print(f'  Original x: {particles.x[0] * 1e3:g} mm')
print(f'  Changed copy x: {particles_copy.x[0] * 1e3:g} mm')

################################################################################
# Choose whether a copied line shares its elements
################################################################################

########################################
# Make shallow and deep copies
########################################
line    = xt.Line(elements = {'qa': quad_a, 'qb': quad_b})
shallow = line.copy(shallow = True)
deep    = line.copy(_context = context)

########################################
# A shallow copy uses the original elements
########################################
shallow['qb'].k1    = CHANGED_K1
print('After changing qb through the shallow copy:')
print(f"  Original line qb k1: {line['qb'].k1:g} m^-2")
print(f"  Shallow copy qb k1: {shallow['qb'].k1:g} m^-2")
print(f"  Deep copy qb k1: {deep['qb'].k1:g} m^-2")

########################################
# A deep copy has independent elements
########################################
deep['qa'].k1   = K1
print('After changing qa through the deep copy:')
print(f"  Original line qa k1: {line['qa'].k1:g} m^-2")
print(f"  Deep copy qa k1: {deep['qa'].k1:g} m^-2")
