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
SOURCE_KQ   = 0.2                       # [m^-2]
TARGET_KQ   = 0.3                       # [m^-2]

########################################
# Slicing
########################################
N_SLICES    = 2

################################################################################
# Build a source line
################################################################################
source = xt.Environment()
source['kq'] = SOURCE_KQ
line = source.new_line(
    name        = 'source_line',
    components  = [
        source.new('quad', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq'),
        source.new('drift', xt.Drift, length = 1.0),
    ],
)

################################################################################
# Import into another environment
################################################################################

########################################
# Keep the target variable values
########################################
target = xt.Environment()
target['kq'] = TARGET_KQ

# Existing target variables win by default. Imported expressions bind to the
# target's variables, so this explicitly chooses a different magnet strength.
target.import_line(line, line_name = 'imported')
line_imported = target['imported']
print('Source k1 [m^-2]:', line['quad'].k1)
print('Imported k1 using target kq [m^-2]:', line_imported['quad'].k1)

########################################
# Adopt the source variable values
########################################
# To adopt source values instead, choose overwrite_vars on a fresh target.
source_values = xt.Environment()
source_values['kq'] = TARGET_KQ
source_values.import_line(line, line_name = 'imported', overwrite_vars = True)
print('Imported k1 using source kq [m^-2]:',
      source_values['imported']['quad'].k1)

################################################################################
# Check independence from the source
################################################################################

########################################
# Change only the target strength
########################################
target['kq'] = 2 * TARGET_KQ
print('After changing target kq:')
print('Source k1 [m^-2]:', line['quad'].k1)
print('Target k1 [m^-2]:', line_imported['quad'].k1)

########################################
# Inspect the imported sequence
########################################
line_imported.get_table().cols['name s element_type'].show()

################################################################################
# Import a sliced line
################################################################################

########################################
# Slice the source quadrupole
########################################
# The earlier imported lines are independent copies. Slice only the source.
line.slice_thick_elements(
    slicing_strategies  = [
        xt.Strategy(
            slicing         = xt.Teapot(N_SLICES),
            element_type    = xt.Quadrupole,
        ),
    ],
)
print('Sliced source; integrated k1l is in m^-1:')
line.get_table(attr = True).cols['name s parent_name k1l'].show()

########################################
# Import slices and their parents
########################################
# The importer carries parent elements and their expressions with the slices.
target_sliced = xt.Environment()
target_sliced.import_line(line, line_name = 'sliced')
line_sliced = target_sliced['sliced']
print('Imported slices and their parents:')
line_sliced.get_table(attr = True).cols['name s parent_name k1l'].show()

########################################
# Change only the imported parent
########################################
target_sliced['kq'] = TARGET_KQ
print('Source parent k1 [m^-2]:', source['quad'].k1)
print('Imported parent k1 [m^-2]:', target_sliced['quad'].k1)

########################################
# Compare source and imported slice strengths
########################################
print('Source slices after changing only the imported kq:')
line.get_table(attr = True).cols['name s parent_name k1l'].show()
print('Imported slices after the change:')
line_sliced.get_table(attr = True).cols['name s parent_name k1l'].show()

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
right_source['k_right'] = TARGET_KQ
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
combined['kq'] = 2 * SOURCE_KQ
for label in ['left', 'right']:
    print(label, 'source/imported parent k1 [m^-2]:',
          collision_lines[label]['quad'].k1,
          combined[label]['quad/' + label].k1)

########################################
# Change only the right source
########################################
right_source['k_right'] = 2 * TARGET_KQ
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
