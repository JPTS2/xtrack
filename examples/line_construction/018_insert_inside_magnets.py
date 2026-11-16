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
K1          = 0.2                       # Quadrupole gradient [m^-2]
K2          = 0.1                       # Sextupole gradient [m^-3]

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
            k1      = K1,
            at      = QUAD_START,
        ),
        env.new('end', xt.Marker, at = LINE_LENGTH),
    ],
)

########################################
# Inspect the starting sequence
########################################
print('Original line\n-------------')
print('The insertion comparisons below each start from this line.')
print('Positions [m].')
table = line.get_table(attr = True)
table.cols['name s_start s_end element_type'].show()
print(f'Original integrated quadrupole strength [m^-1]: {table.k1l.sum():.3f}')
starting_line = line.copy(shallow = True)

################################################################################
# Insert across an occupied thick interval
################################################################################

########################################
# Cover the original quadrupole completely
########################################
# A thick insertion replaces the covered span, including an existing magnet.
# For an equal-length named swap, see 010_replace_elements.py.
env.new('cover', xt.Sextupole, length = 2 * QUAD_LENGTH, k2 = K2)
covered_line = starting_line.copy(shallow = True)
print('\nA thick insertion replaces the entire quadrupole'
      '\n------------------------------------------------')
print('The sextupole covers the quadrupole and adjacent drift space.')
print('It replaces the covered span; the magnetic fields are not added.')
covered_line.insert(
    'cover',
    at      = QUAD_START + QUAD_LENGTH / 2,
    anchor  = 'center',
)
table = covered_line.get_table(attr = True)
table.cols['name s_start s_end element_type'].show()
print(f'Remaining quadrupole k1l [m^-1]: {table.k1l.sum():.3f}')
print(f'Inserted sextupole k2l [m^-2]: {table.k2l.sum():.3f}')
print(f'Total line length [m]: {covered_line.get_length():.3f}')

########################################
# Replace only part of the original magnet
########################################
# The uncovered half remains a thick slice linked to its original parent.
env.new('partial_cover', xt.Sextupole, length = QUAD_LENGTH, k2 = K2)
partial_line = starting_line.copy(shallow = True)
print('\nA partial overlap retains the uncovered half'
      '\n--------------------------------------------')
print('Centre the sextupole on the quadrupole exit: half the quad remains.')
partial_line.insert('partial_cover', anchor = 'center', at = 'quad@end')
table = partial_line.get_table(attr = True)
table.cols[
    'name s_start s_end element_type parent_name'].show(maxwidth = 'full')
print(f'Remaining quadrupole k1l [m^-1]: {table.k1l.sum():.3f}')
print(f'Inserted sextupole k2l [m^-2]: {table.k2l.sum():.3f}')

########################################
# Insert a zero-length marker inside the magnet
########################################
env.new('inside_quad', xt.Marker)
split_line = starting_line.copy(shallow = True)
print('\nA zero-length marker preserves the magnetic body'
      '\n------------------------------------------------')
print('The marker creates a named plane; two thick slices fill the quadrupole.')
split_line.insert('inside_quad', at = 'quad@center')
table = split_line.get_table(attr = True)
table.cols['name s_start s_end element_type'].show(maxwidth = 'full')
print(f'Preserved quadrupole k1l [m^-1]: {table.k1l.sum():.3f}')

########################################
# Remove the marker without merging its neighbouring slices
########################################
# Removing the zero-length marker leaves the slices and geometry in place.
print('\nRemoving the marker leaves the slices in place'
      '\n----------------------------------------------')
print('Removing a marker does not merge the two magnet slices back together.')
without_marker = split_line.copy(shallow = True)
without_marker.remove('inside_quad')
table = without_marker.get_table(attr = True)
table.cols['name s_start s_end element_type'].show(maxwidth = 'full')
print(f'Preserved quadrupole k1l [m^-1]: {table.k1l.sum():.3f}')
print(f'Total line length [m]: {without_marker.get_length():.3f}')
