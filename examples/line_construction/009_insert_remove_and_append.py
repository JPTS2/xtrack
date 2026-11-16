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
print('Starting line\n-------------')
print('Positions [m]. Insertions below occupy drift space.')
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
print('\nInsert a sextupole and replace the quadrupole'
      '\n---------------------------------------------')
print('Insertion splits the drift; equal-length replacement keeps the layout.')
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
print('\nRemove an occurrence, then its unused definition'
      '\n------------------------------------------------')
print('line.remove replaces the sextupole with a drift of the same length.')
print('sext remains in env.elements:', 'sext' in env.elements)

########################################
# Remove the unused environment definition
########################################
# Remove a definition only after no line or expression needs it.
env.remove('sext')
print('After env.remove, sext remains in env.elements:', 'sext' in env.elements)

########################################
# Append a drift
########################################
# Append extends the line; it does not reposition the existing end marker.
env.new('tail', xt.Drift, length = TAIL_LENGTH)
line.append('tail')

########################################
# Inspect the edited geometry
########################################
print('\nAppend a drift after the existing end marker'
      '\n--------------------------------------------')
print('Appending extends the line; the end marker stays at its old position.')
line.get_table().cols['name s_start s_end element_type'].show()
print(f'Edited line length [m]: {line.get_length():.3f}')

################################################################################
# Insert a batch or a prepared subline
################################################################################

########################################
# Insert adjacent elements as a list
########################################
env.new('batch_s1', xt.Sextupole, length = SEXT_LENGTH, k2 = 0.1)
env.new('batch_s2', xt.Sextupole, length = SEXT_LENGTH, k2 = -0.1)
batch_line = starting_line.copy(shallow = True)
print('\nInsert two adjacent elements\n----------------------------')
print(f'Each insertion comparison starts from the original '
      f'{LINE_LENGTH:g} m line.')
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
print('\nInsert a batch using relative anchors'
      '\n-------------------------------------')
print(f'The sextupole ends {SEXT_LENGTH:g} m before the quadrupole entrance.')
print('Both new markers are placed at the new sextupole exit.')
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
print('\nInsert a subline with an internal gap'
      '\n-------------------------------------')
print('Unlike the adjacent list, the subline retains its internal drift.')
subline_line.insert(insert_line, at = SEXT_START, anchor = 'start')
subline_line.get_table().cols['name s_start s_end'].show()

################################################################################
# Remove a single occurrence, all occurrences or a pattern
################################################################################

########################################
# Give the comparison line a repeated quadrupole
########################################
repeated_line = env.new_line(components = ['quad', 'quad', 'batch_s1'])
print('\nCompare removal selections\n--------------------------')
print('Starting sequence [m]; each selection is applied to a separate copy.')
repeated_line.get_table().cols['name s_start s_end'].show()
print('Each removed thick occurrence is replaced by an equal-length drift.')

########################################
# Compare the three selections on separate sequences
########################################
for selection, label in [
    ('quad::1', 'Remove only the second quadrupole occurrence'),
    ('quad', 'Remove every occurrence of quad'),
    ('batch_.*', 'Remove names matching batch_.*'),
]:
    removed_line = repeated_line.copy(shallow = True)
    removed_line.remove(selection)
    print('\n' + label + '\n' + '-' * len(label))
    print('Selection:', selection)
    removed_line.get_table().cols['name s_start s_end'].show()
    print(f'Length after removal [m]: {removed_line.get_length():.3f}')

################################################################################
# Append an object, a list or a prepared subline
################################################################################

########################################
# Append an instantiated element
########################################
object_tail = starting_line.copy(shallow = True)
object_tail.append('new_tail', xt.Drift(length = TAIL_LENGTH))
print('\nAppend a newly instantiated element'
      '\n-----------------------------------')
print(f'Each append comparison starts from the original '
      f'{LINE_LENGTH:g} m line.')
object_tail.get_table().cols['name s_start s_end'].show()

########################################
# Append adjacent elements
########################################
list_tail = starting_line.copy(shallow = True)
list_tail.append(['batch_s1', 'batch_s2'])
print('\nAppend two adjacent elements\n----------------------------')
list_tail.get_table().cols['name s_start s_end'].show()

########################################
# Append the subline with its internal spacing
########################################
subline_tail = starting_line.copy(shallow = True)
subline_tail.append(insert_line)
print('\nAppend a subline with an internal gap'
      '\n-------------------------------------')
print('Appending the subline preserves the gap between its sextupoles.')
subline_tail.get_table().cols['name s_start s_end'].show()
