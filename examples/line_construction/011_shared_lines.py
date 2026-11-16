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
KQ_AFTER    = 0.3                       # Original environment [m^-2]
KQ_COPY     = 0.4                       # Copied environment [m^-2]

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
print('Shared selection and independent environment'
      '\n--------------------------------------------')
print('select() shares element definitions and variables, '
      'but has its own sequence.')
print('env.copy() copies the lines and variables into a separate environment.')
copied_env = env.copy()

########################################
# Inspect the starting strengths
########################################
print(f'Original transport k1 [m^-2]: {line["quad"].k1:.3f}')
print(f'Selected line k1 [m^-2]: {selected["quad"].k1:.3f}')
print(f'Copied transport k1 [m^-2]: '
      f'{copied_env["transport"]["quad"].k1:.3f}')

################################################################################
# Compare shared and independent changes
################################################################################

########################################
# Change the shared strength
########################################
env['kq'] = KQ_AFTER
print('\nChange the original environment variable'
      '\n----------------------------------------')
print(f'Original kq [m^-2]: {env["kq"]:.3f}')
print(f'Original transport k1 [m^-2]: {line["quad"].k1:.3f}')
print(f'Selected line k1 [m^-2]: {selected["quad"].k1:.3f}')
print(f'Unchanged copied transport k1 [m^-2]: '
      f'{copied_env["transport"]["quad"].k1:.3f}')

########################################
# Change the copied environment variable independently
########################################
# The copied expressions remain live, using the copied variables.
copied_env['kq'] = KQ_COPY
print('\nChange the copied environment variable'
      '\n--------------------------------------')
print(f'Copied kq [m^-2]: {copied_env["kq"]:.3f}')
print(f'Original transport k1 [m^-2]: {line["quad"].k1:.3f}')
print(f'Selected line k1 [m^-2]: {selected["quad"].k1:.3f}')
print(f'Copied transport k1 [m^-2]: '
      f'{copied_env["transport"]["quad"].k1:.3f}')

########################################
# Edit only the selected sequence
########################################
selected.remove('end')
print('\nEdit only the selected sequence\n-------------------------------')
print('Remove the end marker from selected; the other sequences stay intact.')
print('Original transport:', line.element_names)
print('Selected line:', selected.element_names)
print('Copied transport:', copied_env['transport'].element_names)
print('Copied selected line:', copied_env['selected'].element_names)
