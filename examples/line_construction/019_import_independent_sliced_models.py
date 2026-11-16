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
print('Prepare independently authored sliced models'
      '\n--------------------------------------------')
print('Use this when two model sections reuse magnet names '
      'in separate sources.')
print('The knobs already have distinct names: kq and k_right.')
print('Left: retain an existing thin-sliced section, '
      'then slice an added magnet.')
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
print('\nRight: slice its two magnets together, using the same element names.')
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
print('\nImport as two named lines in one environment'
      '\n--------------------------------------------')
print('Conflicting magnet and slice names gain /left or /right qualifiers.')
print('parent_name identifies the imported magnet each slice depends on.')
combined = xt.Environment(lines = collision_lines)
for label in ['left', 'right']:
    print('\nImported ' + label + ' magnetic slices; s [m], k1l [m^-1]:')
    imported_table = combined[label].get_table(attr = True)
    magnetic_slices = imported_table.rows[
        (imported_table.element_type == 'ThinSliceQuadrupole')
        | (imported_table.element_type == 'ThickSliceQuadrupole')]
    magnetic_slices.cols[
        'name s element_type parent_name k1l'].show(maxwidth = 'full')
    print(f'Total k1l [m^-1]: {imported_table.k1l.sum():.3f}')
    print('Expected from the two source magnets [m^-1]: '
          f'{2 * QUAD_LENGTH * collision_lines[label]["quad"].k1:.3f}')

################################################################################
# Change imported and source strengths independently
################################################################################

########################################
# Change only the imported left knob
########################################
# The left imported parents use kq; the right imported parents use k_right.
print('\nChange the imported left knob only'
      '\n----------------------------------')
combined['kq'] = 2 * LEFT_KQ
print(f'Combined-environment kq [m^-2]: {combined["kq"]:.3f}')
print('The left source and both right models retain their strengths.')
print(f'{"Model":<10} {"Source k1 [m^-2]":>20} {"Imported k1 [m^-2]":>22}')
for label in ['left', 'right']:
    print(f'{label:<10} {collision_lines[label]["quad"].k1:20.3f}'
          f' {combined[label]["quad/" + label].k1:22.3f}')

########################################
# Change only the right source knob
########################################
print('\nChange the right source knob only'
      '\n---------------------------------')
right_source['k_right'] = 2 * RIGHT_KQ
print(f'Right-source k_right [m^-2]: {right_source["k_right"]:.3f}')
print('The imported right model retains its own variable '
      'and parent definitions.')
print(f'{"Model":<10} {"Source k1 [m^-2]":>20} {"Imported k1 [m^-2]":>22}')
for label in ['left', 'right']:
    print(f'{label:<10} {collision_lines[label]["quad"].k1:20.3f}'
          f' {combined[label]["quad/" + label].k1:22.3f}')

########################################
# Compare the resulting integrated strengths
########################################
print('\nSlice strengths follow the independent changes'
      '\n----------------------------------------------')
print('Sums include all magnetic slices; k1l [m^-1]:')
print(f'{"Model":<10} {"Source sum":>14} {"Imported sum":>16}')
for label in ['left', 'right']:
    source_table    = collision_lines[label].get_table(attr = True)
    imported_table  = combined[label].get_table(attr = True)
    print(f'{label:<10} {source_table.k1l.sum():14.3f}'
          f' {imported_table.k1l.sum():16.3f}')
