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
QUAD_LENGTH     = 0.4                   # [m]
DRIFT_LENGTH    = 1.0                   # [m]
K1_START        = 0.2                   # [m^-2]
K1_SHARED       = 0.3                   # [m^-2]
K1_SINGLE       = 0.5                   # [m^-2]
K1_AFTER        = 0.4                   # [m^-2]
K1_REPLACEMENT  = 0.25                  # [m^-2]

################################################################################
# Reuse one quadrupole three times
################################################################################

########################################
# Build a line with one shared quadrupole
########################################
env = xt.Environment()
env.new('quad', xt.Quadrupole, length = QUAD_LENGTH, k1 = K1_START)
env.new('drift', xt.Drift, length = DRIFT_LENGTH)
env.new('screen.left', xt.Marker)
env.new('screen.right', xt.Marker)
line = env.new_line(
    components  = [
        'screen.left', 'quad', 'drift', 'quad', 'drift', 'quad', 'screen.right',
    ],
)

########################################
# Inspect occurrence names and positions
########################################
# The table adds zero-based :: occurrence suffixes to repeated names.
# env_name identifies the shared element; ::1 identifies its second occurrence.
print('Three occurrences of one quadrupole'
      '\n-----------------------------------')
print('Units: s_start, s_end [m]; k1l [m^-1].')
print('k1l is k1 times the quadrupole length in this example.')
print('env_name identifies the shared definition; '
      '::1 is its second occurrence.')
table = line.get_table(attr = True)
table.cols['name env_name s_start s_end k1l'].show(maxwidth = 'full')

################################################################################
# Change all three quadrupoles, then replace only the second
################################################################################

########################################
# Change the shared element attribute
########################################
# All three occurrences reference the same object, so all three change.
line['quad'].k1 = K1_SHARED
print('\nChange the shared quadrupole strength'
      '\n-------------------------------------')
print(f'Set quad.k1 [m^-2]: {K1_SHARED:.3f}. All three occurrences change.')
table = line.get_table(attr = True)
table.cols['name s_start s_end k1l'].show(maxwidth = 'full')

########################################
# Replace the second occurrence only
########################################
# Create an independent element with the same length to preserve geometry.
env.new(
    'quad_single', xt.Quadrupole,
    length  = QUAD_LENGTH,
    k1      = K1_SINGLE,
)

# Using 'quad' here would replace every occurrence of that environment name.
line.replace('quad::1', 'quad_single')
print('\nReplace only the second occurrence'
      '\n----------------------------------')
print("replace('quad::1', 'quad_single') selects just the second occurrence.")
print('Its replacement has independent attributes and the same length.')
print('Occurrence suffixes are renumbered after the edit; '
      'follow the positions.')
table = line.get_table(attr = True)
table.cols['name env_name s_start s_end k1l'].show(maxwidth = 'full')

################################################################################
# Replace all magnets with a given name or pattern
################################################################################

########################################
# Check which quadrupoles still share their strength
########################################
line['quad'].k1 = K1_AFTER
# Occurrence suffixes are regenerated after edits: the remaining shared quads
# are now quad::0 and quad::1. quad_single keeps its own strength.
print('\nChange the remaining shared quadrupoles'
      '\n---------------------------------------')
print(f'Set quad.k1 [m^-2]: {K1_AFTER:.3f}. '
      'quad_single keeps its own strength.')
table = line.get_table(attr = True)
table.cols['name env_name s_start s_end k1l'].show(maxwidth = 'full')

########################################
# Replace both remaining uses of quad
########################################
env.new(
    'quad_family', xt.Quadrupole,
    length  = QUAD_LENGTH,
    k1      = K1_REPLACEMENT,
)

# Both remaining occurrences are selected; quad_single stays independent.
line.replace('quad', 'quad_family')
print('\nReplace all remaining uses of quad'
      '\n----------------------------------')
print("replace('quad', 'quad_family') selects both remaining uses of quad.")
print('quad_single is a different name and is left unchanged.')
table = line.get_table(attr = True)
table.cols['name env_name s_start s_end k1l'].show(maxwidth = 'full')

########################################
# Replace both screens using a name pattern
########################################
# A regex also selects distinct element names. Here both screens are markers,
# so their common replacement has the same zero length.
env.new('screen_replacement', xt.Marker)
line.replace('screen[.].*', 'screen_replacement')
print('\nReplace names selected by a pattern'
      '\n-----------------------------------')
print("The pattern 'screen[.].*' selects screen.left and screen.right.")
print('Both become occurrences of screen_replacement, a zero-length marker.')
table = line.get_table(attr = True)
table.cols['name env_name s_start s_end k1l'].show(maxwidth = 'full')

################################################################################
# Inspect the preserved geometry
################################################################################
# Every replacement kept its selected element length.
print(f'Final line length [m]: {line.get_length():.3f}')
