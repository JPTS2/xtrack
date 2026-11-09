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

########################################
# Element dimensions and strengths
########################################
DRIFT_LENGTH    = 1.0                   # [m]
QUAD_LENGTH     = 0.4                   # [m]
KQ              = 0.2                   # Quadrupole k1 magnitude [m^-2]

########################################
# Line identity
########################################
LINE_NAME       = 'transport'

################################################################################
# Assemble an ordered sequence
################################################################################

########################################
# Define the elements
########################################
env = xt.Environment()
env.new('start', xt.Marker)
env.new('d1', xt.Drift, length = DRIFT_LENGTH)
env.new('qf', xt.Quadrupole, length = QUAD_LENGTH, k1 = KQ)
env.new('d2', xt.Drift, length = DRIFT_LENGTH)
env.new('qd', xt.Quadrupole, length = QUAD_LENGTH, k1 = -KQ)
env.new('d3', xt.Drift, length = DRIFT_LENGTH)
env.new('end', xt.Marker)

########################################
# Place consecutive components
########################################
# Without explicit positions, elements follow one another without gaps.
line = env.new_line(
    name        = LINE_NAME,
    components  = ['start', 'd1', 'qf', 'd2', 'qd', 'd3', 'end'],
)

################################################################################
# Inspect the assembled line
################################################################################

########################################
# Read positions and length
########################################
table = line.get_table()
table.cols['name s_start s_center s_end element_type'].show()
print('Total line length [m]:', line.get_length())

########################################
# Retrieve the registered line
########################################
line_registered = env[LINE_NAME]
print('Registered line names:', list(env.lines.keys()))
print('Registered sequence:', line_registered.element_names)
print('Same line object:', line_registered is line)
