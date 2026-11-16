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
print('Default import keeps existing target variables'
      '\n----------------------------------------------')
print('Imported expressions use variables in the target environment.')
print(f'Source kq [m^-2]: {source["kq"]:.3f}')
print(f'Target kq before import [m^-2]: {target["kq"]:.3f}')
target.import_line(line, line_name = 'imported')
line_imported = target['imported']
print(f'Source quad.k1 [m^-2]: {line["quad"].k1:.3f}')
print(f'Imported quad.k1 [m^-2]: {line_imported["quad"].k1:.3f}')

########################################
# Adopt the source variable values
########################################
# To adopt source values instead, choose overwrite_vars on a fresh target.
target_overwrite = xt.Environment()
target_overwrite['kq'] = TARGET_KQ
print('\nChoose source values with overwrite_vars'
      '\n----------------------------------------')
print('Use a fresh target to compare the two import policies.')
print(f'Target kq before import [m^-2]: {target_overwrite["kq"]:.3f}')
target_overwrite.import_line(
    line,
    line_name       = 'imported',
    overwrite_vars  = True,
)
print(f'Target kq after import [m^-2]: {target_overwrite["kq"]:.3f}')
print(f'Imported quad.k1 [m^-2]: '
      f'{target_overwrite["imported"]["quad"].k1:.3f}')

################################################################################
# Check independence from the source
################################################################################

########################################
# Change only the target strength
########################################
target['kq'] = 2 * TARGET_KQ
print('\nChange only the first target variable'
      '\n-------------------------------------')
print(f'Target kq [m^-2]: {target["kq"]:.3f}')
print(f'Source quad.k1 [m^-2]: {line["quad"].k1:.3f}')
print(f'Target quad.k1 [m^-2]: {line_imported["quad"].k1:.3f}')
print(f'Other target quad.k1 [m^-2]: '
      f'{target_overwrite["imported"]["quad"].k1:.3f}')

########################################
# Inspect the imported sequence
########################################
print('\nImported element sequence\n-------------------------')
print('s [m]')
line_imported.get_table().cols['name s element_type'].show()

################################################################################
# Import a sliced line
################################################################################

########################################
# Slice the source quadrupole
########################################
# A sliced tracking model still needs its parents to define its strengths.
# The earlier imported lines are independent copies. Slice only the source.
print('\nSlice only the source quadrupole'
      '\n--------------------------------')
print('Thin slices refer to a parent magnet; '
      'importing them must carry that parent.')
line.slice_thick_elements(
    slicing_strategies  = [
        xt.Strategy(
            slicing         = xt.Teapot(N_SLICES),
            element_type    = xt.Quadrupole,
        ),
    ],
)
print('Units: s [m], k1l [m^-1].')
line.get_table(attr = True).cols[
    'name s parent_name k1l'].show(maxwidth = 'full')

########################################
# Import slices and their parents
########################################
# The importer carries parent elements and their expressions with the slices.
target_sliced = xt.Environment()
print('\nImport slices together with their parent'
      '\n----------------------------------------')
print('The empty target receives the parent '
      'and its strength-variable dependency.')
target_sliced.import_line(line, line_name = 'sliced')
line_sliced = target_sliced['sliced']
print('Imported sequence:')
line_sliced.get_table(attr = True).cols[
    'name s parent_name k1l'].show(maxwidth = 'full')

########################################
# Change only the imported parent
########################################
print('\nChange the imported parent through its variable'
      '\n-----------------------------------------------')
print('The imported parent and slices respond; the source stays unchanged.')
target_sliced['kq'] = TARGET_KQ
print(f'Source parent k1 [m^-2]: {source["quad"].k1:.3f}')
print(f'Imported parent k1 [m^-2]: {target_sliced["quad"].k1:.3f}')

########################################
# Compare source and imported slice strengths
########################################
print('\nSource slices after changing only the imported kq:')
line.get_table(attr = True).cols[
    'name s parent_name k1l'].show(maxwidth = 'full')
print('\nImported slices after the change:')
line_sliced.get_table(attr = True).cols[
    'name s parent_name k1l'].show(maxwidth = 'full')
