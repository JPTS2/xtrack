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
# Model and strengths
########################################
QUAD_LENGTH = 0.4                       # [m]
LEFT_KQ     = 0.2                       # [m^-2]
RIGHT_KQ    = 0.3                       # [m^-2]

########################################
# Slicing
########################################
N_SLICES    = 2

################################################################################
# Build a source line
################################################################################
source = xt.Environment()
source['kq'] = LEFT_KQ
line = source.new_line(
    name        = 'source_line',
    components  = [
        source.new('quad', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq'),
        source.new('drift', xt.Drift, length = 1.0),
    ],
)

################################################################################
# Slice the first source quadrupole
################################################################################
line.slice_thick_elements(slicing_strategies = [
    xt.Strategy(slicing = xt.Teapot(N_SLICES), element_type = xt.Quadrupole),
])

################################################################################
# Import two sliced lines with the same element names
################################################################################

########################################
# Add a thick-sliced quadrupole to the first source
########################################
source.new(
    'quad_thick', xt.Quadrupole,
    length  = QUAD_LENGTH,
    k1      = 'kq',
)
left = source.new_line(components = [line, 'quad_thick'])
left.slice_thick_elements(
    slicing_strategies  = [
        xt.Strategy(slicing = None),  # Keep the existing thin slices.
        xt.Strategy(
            slicing = xt.Teapot(N_SLICES, mode = 'thick'),
            name    = 'quad_thick',
        ),
    ],
)

########################################
# Build the second source with its own strength variable
########################################
# Reuse the element names, but give this source its own knob, k_right.
right_source = xt.Environment()
right_source['k_right'] = RIGHT_KQ
right = right_source.new_line(
    components  = [
        right_source.new(
            'quad', xt.Quadrupole,
            length  = QUAD_LENGTH,
            k1      = 'k_right',
        ),
        right_source.new('drift', xt.Drift, length = 1.0),
        right_source.new(
            'quad_thick', xt.Quadrupole,
            length  = QUAD_LENGTH,
            k1      = 'k_right',
        ),
    ],
)
right.slice_thick_elements(
    slicing_strategies  = [
        xt.Strategy(
            slicing = xt.Teapot(N_SLICES, mode = 'thin'),
            name    = 'quad',
        ),
        xt.Strategy(
            slicing = xt.Teapot(N_SLICES, mode = 'thick'),
            name    = 'quad_thick',
        ),
    ],
)
collision_lines = {'left': left, 'right': right}

########################################
# Import together and inspect the renamed parents
########################################
# Both sources use quad and quad_thick. Importing together adds /left and
# /right to their names; parent_name shows which parent each slice uses.
combined = xt.Environment(lines = collision_lines)
for label in ['left', 'right']:
    print('Imported', label, 'line; s [m], integrated k1l [m^-1]:')
    imported_table = combined[label].get_table(attr = True)
    imported_table.cols['name s element_type parent_name k1l'].show()
    print('Integrated k1l sum [m^-1]:', sum(imported_table.k1l))
    print('Expected k1l sum [m^-1]:',
          2 * QUAD_LENGTH * collision_lines[label]['quad'].k1)

################################################################################
# Change imported and source strengths independently
################################################################################

########################################
# Change only the imported left line
########################################
# Both imported left parents follow kq; the right parents follow k_right.
combined['kq'] = 2 * LEFT_KQ
for label in ['left', 'right']:
    print(label, 'source/imported parent k1 [m^-2]:',
          collision_lines[label]['quad'].k1,
          combined[label]['quad/' + label].k1)

########################################
# Change only the right source
########################################
right_source['k_right'] = 2 * RIGHT_KQ
for label in ['left', 'right']:
    print(label, 'source/imported parent k1 [m^-2]:',
          collision_lines[label]['quad'].k1,
          combined[label]['quad/' + label].k1)

########################################
# Compare the resulting slice strengths
########################################
for label in ['left', 'right']:
    source_table    = collision_lines[label].get_table(attr = True)
    imported_table  = combined[label].get_table(attr = True)
    print('Source', label, 'after independent changes:')
    source_table.cols['name s parent_name k1l'].show()
    print('Source integrated k1l sum [m^-1]:', sum(source_table.k1l))
    print('Imported', label, 'after independent changes:')
    imported_table.cols['name s parent_name k1l'].show()
    print('Imported integrated k1l sum [m^-1]:', sum(imported_table.k1l))
