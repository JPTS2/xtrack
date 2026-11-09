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
KQ_START    = 0.2                       # [m^-2]
KQ_AFTER    = 0.3                       # [m^-2]

################################################################################
# Build lines in one environment
################################################################################

########################################
# Define the original line
########################################
env = xt.Environment()
env['kq'] = KQ_START
line = env.new_line(
    name        = 'transport',
    components  = [
        env.new('start', xt.Marker),
        env.new('quad', xt.Quadrupole, length = 0.4, k1 = 'kq'),
        env.new('drift', xt.Drift, length = 1.0),
        env.new('end', xt.Marker),
    ],
)

########################################
# Create shared and independent cases
########################################
# select creates another line in the same environment, sharing its elements
# and variables. It has a separate element sequence that can be edited.
selected = line.select(start = 'quad', name = 'selected')
independent = env.copy()

########################################
# Inspect the starting strengths
########################################
print('Starting k1 [m^-2]:', line['quad'].k1, selected['quad'].k1,
      independent['transport']['quad'].k1)

################################################################################
# Compare shared and independent changes
################################################################################

########################################
# Change the shared strength
########################################
env['kq'] = KQ_AFTER
print('Changed k1 [m^-2]:', line['quad'].k1, selected['quad'].k1,
      independent['transport']['quad'].k1)

########################################
# Edit only the selected sequence
########################################
selected.remove('end')
print('Original sequence:', line.element_names)
print('Selected sequence:', selected.element_names)
print('Independent sequence:', independent['transport'].element_names)
