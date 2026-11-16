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
QUAD_LENGTH     = 0.3                   # [m]
KQ_START        = 0.2                   # [m^-2]
KQ_AFTER        = 0.3                   # [m^-2]
ROLL_START      = 1.0e-3                # [rad]
ROLL_AFTER      = 2.0e-3                # [rad]
DRIFT_LENGTH    = 3.0                   # [m]
DRIFT_AFTER     = 4.0                   # [m]
ROLL_SINGLE     = 3.0e-3                # One converted occurrence [rad]

################################################################################
# Create repeated elements, a replica and a clone
################################################################################

########################################
# Define the original, replica and clone
########################################
env = xt.Environment()
env['kq'] = KQ_START
env.new('quad', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq')
env.new('replica', 'quad', mode = 'replica')
env.new('clone', 'quad', mode = 'clone')
line = env.new_line(components = ['quad', 'quad', 'replica', 'clone'])

########################################
# Compare starting strengths and rolls
########################################
# Repeated occurrences use one named element. A replica references that
# element under a new name. A clone has separate attributes and copies the
# strength expressions. The clone was created before rolling the original.
line['quad'].rot_s_rad = ROLL_START
print('Repeated occurrences, a replica and a clone'
      '\n-------------------------------------------')
print('Repeated quad occurrences and replica follow the original definition.')
print('clone keeps copied attributes and expressions; only quad was rolled.')
print('Units: s [m], k1l [m^-1], rot_s_rad [rad].')
line.get_table(attr = True).cols['name s k1l rot_s_rad isreplica'].show()

################################################################################
# Compare inherited expressions and independent attributes
################################################################################

########################################
# Change the family strength and original roll
########################################
env['kq'] = KQ_AFTER
line['quad'].rot_s_rad = ROLL_AFTER
print('\nChange the shared strength and original roll'
      '\n--------------------------------------------')
print(f'kq [m^-2]: {env["kq"]:.3f}; original roll [rad]: {ROLL_AFTER:.3f}')
print('All strengths follow kq. The clone keeps its independent roll.')
line.get_table(attr = True).cols['name s k1l rot_s_rad isreplica'].show()

########################################
# Replace only the original strength expression
########################################
# Replacing the original expression does not replace the clone's expression.
line['quad'].k1 = '2 * kq'
print('\nReplace the original strength expression'
      '\n----------------------------------------')
print('quad.k1 is now 2 * kq; the replica follows the original.')
print('The clone keeps its copied kq expression.')

# The table resolves replica attributes through their parent. Direct access
# to line['replica'] returns the Replica object, not its parent quadrupole.
line.get_table(attr = True).cols['name s k1l'].show()

################################################################################
# Give each occurrence an independent element
################################################################################

########################################
# Replace repeated elements and replicas
########################################
print('\nConvert references into independent elements'
      '\n--------------------------------------------')
print('Replace repeated occurrences, then resolve replicas into full elements.')
line.replace_all_repeated_elements()
line.replace_all_replicas()

########################################
# Inspect the updated element table
########################################
line.get_table(attr = True).cols['name s k1l rot_s_rad isreplica'].show()

########################################
# Change one converted occurrence
########################################
first_quad_name = line.element_names[0]
line[first_quad_name].rot_s_rad = ROLL_SINGLE
print('\nChange one converted element\n----------------------------')
print(f'Only {first_quad_name} receives roll [rad]: {ROLL_SINGLE:.3f}')
line.get_table(attr = True).cols['name s k1l rot_s_rad isreplica'].show()

########################################
# Check the shared strength variable
########################################
env['kq'] = KQ_START
print('\nIndependent attributes, shared strength variables'
      '\n-------------------------------------------------')
print(f'Change kq [m^-2]: {env["kq"]:.3f}')
print('Converting the elements preserved their strength expressions.')
line.get_table(attr = True).cols['name k1l rot_s_rad'].show()

################################################################################
# Choose whether generated drifts become independent
################################################################################

########################################
# Compare default and explicit drift conversion
########################################
# Equal gaps reuse one generated drift. Start from separate environments so
# changing either converted line cannot change the other comparison.
gaps = env.new_line(
    length      = 3 * DRIFT_LENGTH,
    components  = [
        env.new('marker', xt.Marker, at = DRIFT_LENGTH),
        env.place('marker', at = 2 * DRIFT_LENGTH),
    ],
)
shared_drifts = gaps.copy()
separate_drifts = gaps.copy()
shared_drifts.replace_all_repeated_elements()
separate_drifts.replace_all_repeated_elements(replace_generated_drifts = True)
print('\nGenerated drifts stay shared by default'
      '\n---------------------------------------')
print('Equal gaps initially reuse one drift definition; s_start, s_end [m].')
print('Default replace_all_repeated_elements() leaves that sharing intact.')
shared_drifts.get_table().cols['name env_name s_start s_end'].show()
print('\nConvert generated drifts explicitly'
      '\n-----------------------------------')
print('Use this when one gap needs to change without changing the others.')
print('replace_generated_drifts = True gives each gap its own drift element.')
separate_drifts.get_table().cols['name env_name s_start s_end'].show()

########################################
# Change one independent drift
########################################
print('\nChange only one independent drift'
      '\n---------------------------------')
first_drift = separate_drifts.element_names[0]
separate_drifts[first_drift].length = DRIFT_AFTER
print(f'Updated {first_drift} length [m]: {DRIFT_AFTER:.3f}')
print('Default first and second gaps share an object:',
      shared_drifts.get(shared_drifts.element_names[0])
      is shared_drifts.get(shared_drifts.element_names[2]))
print('Converted first and second gaps share an object:',
      separate_drifts.get(separate_drifts.element_names[0])
      is separate_drifts.get(separate_drifts.element_names[2]))
separate_drifts.get_table().cols['name s_start s_end'].show()
print(f'Unchanged default line length [m]: {shared_drifts.get_length():.3f}')
