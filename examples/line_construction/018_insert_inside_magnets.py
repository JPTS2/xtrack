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
starting_line = line.copy(shallow = True)

################################################################################
# Insert across an occupied thick interval
################################################################################

########################################
# Cover the original quadrupole completely
########################################
# A thick insertion replaces the covered span, including an existing magnet.
# Use line.replace for an equal-length named swap;009 shows that operation.
env.new('cover', xt.Sextupole, length = 2 * QUAD_LENGTH, k2 = 0.1)
covered_line = starting_line.copy(shallow = True)
covered_line.insert(
    'cover',
    at      = QUAD_START + QUAD_LENGTH / 2,
    anchor  = 'center',
)
covered_line.get_table().cols['name s_start s_end element_type'].show()
print('Covered line length [m]:', covered_line.get_length())

########################################
# Replace only part of the original magnet
########################################
# The uncovered half remains a thick slice linked to its original parent.
env.new('partial_cover', xt.Sextupole, length = QUAD_LENGTH, k2 = 0.1)
partial_line = starting_line.copy(shallow = True)
partial_line.insert('partial_cover', anchor = 'center', at = 'quad@end')
partial_line.get_table().cols[
    'name s_start s_end element_type parent_name'].show()

########################################
# Compare a thin insertion inside the magnet
########################################
env.new('inside_quad', xt.Marker)
split_line = starting_line.copy(shallow = True)
split_line.insert('inside_quad', at = 'quad@center')
split_line.get_table().cols['name s_start s_end element_type'].show()
# Removing the zero-length marker leaves the slices and geometry in place.
without_marker = split_line.copy(shallow = True)
without_marker.remove('inside_quad')
without_marker.get_table().cols['name s_start s_end element_type'].show()
