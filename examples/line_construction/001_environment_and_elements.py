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
QUAD_LENGTH = 0.4                       # [m]
KQ_START    = 0.2                       # [m^-2]
KQ_AFTER    = 0.3                       # [m^-2]

################################################################################
# Define named elements
################################################################################

########################################
# Create an environment
########################################
# An environment holds named elements, variables and lines. Creating an
# element does not yet put it at a position in a line.
env = xt.Environment()

########################################
# Register element definitions
########################################
env.new('quad', xt.Quadrupole, length = QUAD_LENGTH, k1 = KQ_START)
env.new('screen', xt.Marker)
print('Registered element names:', list(env.elements.keys()))

################################################################################
# Inspect and change an element
################################################################################

########################################
# Read the named definition
########################################
print('Element class:', type(env.get('quad')).__name__)
print('Quadrupole length [m]:', env['quad'].length)
print('Initial quadrupole k1 [m^-2]:', env['quad'].k1)
env.info('quad')

########################################
# Change a numerical attribute
########################################
# Here k1 is a number. Strength expressions are introduced separately.
env['quad'].k1 = KQ_AFTER
print('Updated quadrupole k1 [m^-2]:', env['quad'].k1)
print('Registered element names after editing:', list(env.elements.keys()))
