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
starting_line = line.copy(shallow = True)

################################################################################
# Insert and replace elements
################################################################################

########################################
# Insert into a drift
########################################
# Inserting into a drift splits that drift while preserving the line length.
sextupole = xt.Sextupole(length = SEXT_LENGTH, k2 = 0.1)
line.insert(
    'sext', sextupole,
    at      = SEXT_START,
    anchor  = 'start',
)

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
print('Removed from the line, still defined:', 'sext' in env.elements)

########################################
# Remove the unused environment definition
########################################
# Remove a definition only after no line or expression needs it.
env.remove('sext')
print('After removing its definition:', 'sext' in env.elements)

########################################
# Append a drift
########################################
# Append extends the line; it does not reposition the existing end marker.
env.new('tail', xt.Drift, length = TAIL_LENGTH)
line.append('tail')

########################################
# Inspect the edited geometry
########################################
print('After removing sext and appending the tail:')
line.get_table().cols['name s_start s_end element_type'].show()
print(f'Edited line length: {line.get_length():.3f} m')

################################################################################
# Insert a batch or a prepared subline
################################################################################

########################################
# Insert adjacent elements as a list
########################################
env.new('batch_s1', xt.Sextupole, length = SEXT_LENGTH, k2 = 0.1)
env.new('batch_s2', xt.Sextupole, length = SEXT_LENGTH, k2 = -0.1)
batch_line = starting_line.copy(shallow = True)
batch_line.insert(
    ['batch_s1', 'batch_s2'],
    at      = SEXT_START,
    anchor  = 'start',
)
batch_line.get_table().cols['name s_start s_end'].show()

########################################
# Place a batch relative to existing and new elements
########################################
anchor_line = starting_line.copy(shallow = True)
anchor_line.insert([
    env.place(
        'batch_s1',
        anchor  = 'end',
        at      = -SEXT_LENGTH,
        from_   = 'quad@start',
    ),
    env.new('batch_m1', xt.Marker, at = 'batch_s1@end'),
    env.new('batch_m2', xt.Marker, at = 'batch_s1@end'),
])
print('Two markers share the new sextupole exit:')
anchor_line.get_table().cols['name s_start s_end'].show()

########################################
# Retain the spacing of a prepared subline
########################################
insert_line = env.new_line(
    components  = [
        'batch_s1',
        env.place('batch_s2', at = 2 * SEXT_LENGTH, anchor = 'start'),
    ],
)
subline_line = starting_line.copy(shallow = True)
subline_line.insert(insert_line, at = SEXT_START, anchor = 'start')
subline_line.get_table().cols['name s_start s_end'].show()

################################################################################
# Insert across an occupied thick interval
################################################################################

########################################
# Cover the original quadrupole completely
########################################
# A thick insertion replaces the covered span, including an existing magnet.
# The equal-length replace operation above is preferable for a named swap.
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

################################################################################
# Remove a single occurrence, all occurrences or a pattern
################################################################################

########################################
# Give the comparison line a repeated quadrupole
########################################
repeated_line = env.new_line(components = ['quad', 'quad', 'batch_s1'])
repeated_line.get_table().cols['name s_start s_end'].show()

########################################
# Compare the three selections on separate sequences
########################################
for selection in ['quad::1', 'quad', 'batch_.*']:
    removed_line = repeated_line.copy(shallow = True)
    removed_line.remove(selection)
    print('Removed:', selection)
    removed_line.get_table().cols['name s_start s_end'].show()
    print('Length after removal [m]:', removed_line.get_length())

################################################################################
# Append an object, a list or a prepared subline
################################################################################

########################################
# Append an instantiated element
########################################
object_tail = starting_line.copy(shallow = True)
object_tail.append('new_tail', xt.Drift(length = TAIL_LENGTH))
object_tail.get_table().cols['name s_start s_end'].show()

########################################
# Append adjacent elements
########################################
list_tail = starting_line.copy(shallow = True)
list_tail.append(['batch_s1', 'batch_s2'])
list_tail.get_table().cols['name s_start s_end'].show()

########################################
# Append the subline with its internal spacing
########################################
subline_tail = starting_line.copy(shallow = True)
subline_tail.append(insert_line)
subline_tail.get_table().cols['name s_start s_end'].show()
