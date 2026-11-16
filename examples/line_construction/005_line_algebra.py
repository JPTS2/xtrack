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
QUAD_LENGTH     = 0.3                   # [m]
DRIFT_LENGTH    = 1.0                   # [m]
KQ              = 0.2                   # Quadrupole k1 magnitude [m^-2]
N_REPEATS       = 3

################################################################################
# Build a short section
################################################################################
env = xt.Environment()
section = env.new_line(
    components  = [
        env.new('qf', xt.Quadrupole, length = QUAD_LENGTH, k1 = KQ),
        env.new('drift', xt.Drift, length = DRIFT_LENGTH),
        env.new('qd', xt.Quadrupole, length = QUAD_LENGTH, k1 = -KQ),
    ],
)

################################################################################
# Combine existing lines
################################################################################

########################################
# Reverse and concatenate
########################################
# Unary minus reverses element order. It does not invert magnet strengths,
# backtrack particles or turn this straight geometry into a closed ring.
symmetric_section = section + (-section)

########################################
# Repeat the combined section
########################################
# Repeated occurrences share the named element definitions.
line = N_REPEATS * symmetric_section

################################################################################
# Inspect the resulting sequences
################################################################################

########################################
# Original and reversed sections
########################################
print('Original section\n----------------')
print('s [m]; k1l [m^-1]')
section.get_table(attr = True).cols['name s element_type k1l'].show()
print('\nReversed section\n----------------')
print('-section reverses element order and preserves the magnet strengths.')
(-section).get_table(attr = True).cols['name s element_type k1l'].show()

########################################
# Concatenated and repeated sections
########################################
print('\nOriginal plus reversed section\n------------------------------')
print('section + (-section) concatenates the two assembled sequences.')
symmetric_section.get_table(attr = True).cols[
    'name s element_type k1l'].show()
print('\nRepeated section\n----------------')
print(f'Repetitions: {N_REPEATS}')
print('Repeated occurrences share the original named element definitions.')
line.get_table(attr = True).cols['name s element_type k1l'].show()
print(f'Total length [m]: {line.get_length():.3f}')
