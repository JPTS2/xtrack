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
KQ_START        = 0.2                   # [m^-2]
KQ_AFTER        = 0.3                   # [m^-2]
TILT_START      = 1.0e-3                # [rad]
TILT_AFTER      = 2.0e-3                # [rad]
DRIFT_LENGTH    = 3.0                   # [m]
DRIFT_AFTER     = 4.0                   # [m]
TILT_SINGLE     = 3.0e-3                # One converted occurrence [rad]

################################################################################
# Create repeated elements, a replica and a clone
################################################################################

########################################
# Define the original, replica and clone
########################################
env = xt.Environment()
env['kq'] = KQ_START
env.new('quad', xt.Quadrupole, length = 0.3, k1 = 'kq')
env.new('replica', 'quad', mode = 'replica')
env.new('clone', 'quad', mode = 'clone')
line = env.new_line(components = ['quad', 'quad', 'replica', 'clone'])

########################################
# Compare starting strengths and tilts
########################################
# Repeated occurrences share one object. A replica gives that same object a
# different name. A clone has its own attributes and inherits expressions.
line['quad'].rot_s_rad = TILT_START
print('Starting integrated strengths and tilts:')
line.get_table(attr = True).cols['name s k1l rot_s_rad isreplica'].show()

################################################################################
# Compare inherited expressions and independent attributes
################################################################################

########################################
# Change the family strength and original tilt
########################################
env['kq'] = KQ_AFTER
line['quad'].rot_s_rad = TILT_AFTER
print('All strengths follow kq; only the original and replica share a tilt:')
line.get_table(attr = True).cols['name s k1l rot_s_rad isreplica'].show()

########################################
# Replace only the original strength expression
########################################
# Replacing the original expression does not replace the clone's expression.
line['quad'].k1 = '2 * kq'
print('Original and replica change; the clone keeps its expression:')

# The table resolves replica attributes through their parent. Direct access
# to line['replica'] returns the Replica object, not its parent quadrupole.
line.get_table(attr = True).cols['name s k1l'].show()

################################################################################
# Give each occurrence an independent element
################################################################################

########################################
# Replace repeated elements and replicas
########################################
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
line[first_quad_name].rot_s_rad = TILT_SINGLE
print(f'Only {first_quad_name} receives the new tilt:')
line.get_table(attr = True).cols['name s k1l rot_s_rad isreplica'].show()

########################################
# Check the shared strength variable
########################################
env['kq'] = KQ_START
print('The converted elements still follow the shared strength variable:')
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
print('Default conversion retains shared generated drifts:')
shared_drifts.get_table().cols['name env_name s_start s_end'].show()
print('Explicit conversion gives each gap its own drift:')
separate_drifts.get_table().cols['name env_name s_start s_end'].show()

########################################
# Change one independent drift
########################################
first_drift = separate_drifts.element_names[0]
separate_drifts[first_drift].length = DRIFT_AFTER
print('Default first and second gaps share an object:',
      shared_drifts.get(shared_drifts.element_names[0])
      is shared_drifts.get(shared_drifts.element_names[2]))
print('Converted first and second gaps share an object:',
      separate_drifts.get(separate_drifts.element_names[0])
      is separate_drifts.get(separate_drifts.element_names[2]))
separate_drifts.get_table().cols['name s_start s_end'].show()
