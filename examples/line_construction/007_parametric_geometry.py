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
SCALE_START = 1.0
SCALE_AFTER = 1.5

################################################################################
# Define geometry expressions
################################################################################

########################################
# Define element lengths
########################################
env = xt.Environment()
env['scale'] = SCALE_START

env.new('qf', xt.Quadrupole, length = '0.4 * scale', k1 = 0.2)
env.new('qd', xt.Quadrupole, length = '0.4 * scale', k1 = -0.2)

########################################
# Retain placement expressions in compose mode
########################################
# Compose mode retains placement expressions, not just generated drift lengths.
line = env.new_line(
    name    = 'parametric_line',
    length  = '5 * scale',
    refer   = 'start',
    compose = True,
)
line.place('qf', at = 'scale')
line.place('qd', at = '2 * scale', from_ = 'qf@end')

########################################
# Assemble and inspect the starting geometry
########################################
line.end_compose()

table_before = line.get_table()
print('Before changing the geometry scale:')
table_before.cols['name s_start s_end'].show()

################################################################################
# Regenerate the geometry
################################################################################

########################################
# Change the scale and rebuild
########################################
# Element attributes update through their expressions when variables change.
# Re-enter compose mode to recompute the layout from the saved placements.
env['scale'] = SCALE_AFTER
line.regenerate_from_composer()
line.end_compose()

########################################
# Inspect the regenerated positions
########################################
table = line.get_table()
print('After regenerating:')
table.cols['name s_start s_end'].show()
print(f'Length: {table_before.s[-1]:.3f} -> {table.s[-1]:.3f} m')

################################################################################
# Compose reversed and repeated sublines
################################################################################

########################################
# Combine the retained geometry
########################################
# Minus reverses the element order; it is not a beam-direction transformation.
composite = env.new_line(
    name        = 'composite',
    compose     = True,
    components  = [line, -line, 2 * line],
)
composite.end_compose()
print('Forward, reversed, then two forward sublines:')
composite.get_table().cols['name s_start s_end'].show()

########################################
# Restore and regenerate the placed subline
########################################
restored = xt.Line.from_dict(line.to_dict())
restored['scale'] = SCALE_START
restored.regenerate_from_composer()
restored.end_compose()
print('Restored placed subline after resetting the scale:')
restored.get_table().cols['name s_start s_end'].show()
print('Original and restored subline lengths [m]:',
      line.get_length(), restored.get_length())
