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
N_CELLS         = 3

################################################################################
# Build a half cell
################################################################################
env = xt.Environment()
half_cell = env.new_line(
    components  = [
        env.new('qf', xt.Quadrupole, length = QUAD_LENGTH, k1 = 0.2),
        env.new('drift', xt.Drift, length = DRIFT_LENGTH),
        env.new('qd', xt.Quadrupole, length = QUAD_LENGTH, k1 = -0.2),
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
cell = half_cell + (-half_cell)

########################################
# Repeat the cell
########################################
line = N_CELLS * cell

########################################
# Inspect the original and reversed sequences
########################################
print('Half cell:')
half_cell.get_table().cols['name s element_type'].show()
print('Reversed half cell:')
(-half_cell).get_table().cols['name s element_type'].show()

########################################
# Inspect the repeated line
########################################
print('Repeated line:')
line.get_table().cols['name s element_type'].show(maxrows = 16)
print(f'Length: {line.get_length():.3f} m')
