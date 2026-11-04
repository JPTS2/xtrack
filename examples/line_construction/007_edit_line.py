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
LINE_LENGTH = 6.0                       # [m]
QUAD_LENGTH = 0.4                       # [m]
QUAD_START  = 3.0                       # [m]
SEXT_START  = 1.0                       # [m]
SEXT_LENGTH = 0.2                       # [m]
TAIL_LENGTH = 0.5                       # [m]

################################################################################
# Build the starting line
################################################################################

########################################
# Define the starting line
########################################
env = xt.Environment()
line = env.new_line(
    length      = LINE_LENGTH,
    refer       = 'start',
    components  = [
        env.new(
            'quad', xt.Quadrupole,
            length  = QUAD_LENGTH,
            k1      = 0.2,
            at      = QUAD_START,
        ),
        env.new('end', xt.Marker, at = LINE_LENGTH),
    ],
)

########################################
# Inspect the starting sequence
########################################
print('Before editing:')
line.get_table().cols['name s_start s_end element_type'].show()

################################################################################
# Insert and replace elements
################################################################################

########################################
# Insert into a drift
########################################
# Inserting into a drift splits that drift while preserving the line length.
env.new('sext', xt.Sextupole, length = SEXT_LENGTH, k2 = 0.1)
line.insert(env.place('sext', at = SEXT_START, anchor = 'start'))

########################################
# Replace an equal-length element
########################################
# Replacement requires equal lengths.
env.new(
    'quad_replacement', xt.Quadrupole,
    length  = QUAD_LENGTH,
    k1      = 0.3,
)
line.replace('quad', 'quad_replacement')

########################################
# Inspect the changed sequence
########################################
print('After insertion and replacement:')
line.get_table().cols['name s_start s_end element_type'].show()

################################################################################
# Remove and append elements
################################################################################

########################################
# Remove the sextupole
########################################
# Removing a thick element replaces its occupied length with a drift.
# It removes the occurrence from this line, not the object from the environment.
line.remove('sext')

########################################
# Append a drift
########################################
# Append extends the line; it does not reposition the existing end marker.
env.new('tail', xt.Drift, length = TAIL_LENGTH)
line.append('tail')

########################################
# Inspect the final geometry
########################################
print('After removing sext and appending the tail:')
line.get_table().cols['name s_start s_end element_type'].show()
print(f'Final length: {line.get_length():.3f} m')
print(f'Sextupole still in the environment: {"sext" in env.elements}')
