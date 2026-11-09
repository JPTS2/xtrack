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
# Element dimensions and strengths
########################################
DRIFT_LENGTH    = 1.0                   # [m]
QUAD_LENGTH     = 1.0                   # [m]
KQ_START        = 0.2                   # [m^-2]
KQ_AFTER        = 0.3                   # [m^-2]

########################################
# Cut positions
########################################
# Positions from the line entrance, in metres.
CUT_POSITIONS   = [0.5, 1.25, 1.75]

################################################################################
# Build the original line
################################################################################
env = xt.Environment()
env['kq'] = KQ_START
line = env.new_line(
    components  = [
        env.new('d1', xt.Drift, length = DRIFT_LENGTH),
        env.new('quad', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq'),
        env.new('d2', xt.Drift, length = DRIFT_LENGTH),
    ],
)
table_before = line.get_table()
table_before.cols['name s_start s_end element_type'].show()
print('Original length [m]:', line.get_length())

################################################################################
# Cut at the requested positions
################################################################################

########################################
# Create thick slices
########################################
# This changes the line in place. Cuts inside an element create slices;
# already existing boundaries need no extra cut.
line.cut_at_s(CUT_POSITIONS)
table = line.get_table(attr = True)
table.cols['name s_start s_end element_type parent_name'].show()
print('Length after cutting [m]:', line.get_length())

########################################
# Inspect the requested boundaries
########################################
for position in CUT_POSITIONS:
    nearest = min(table.s_start, key = lambda s: abs(s - position))
    print(f'Requested cut: {position:.3f} m; boundary: {nearest:.3f} m')

################################################################################
# Change the parent quadrupole strength
################################################################################

########################################
# Inspect the original slice strengths
########################################
# Thick slices retain their parent element; they are not independent magnets.
quad_slices = table.rows[table.parent_name == 'quad']
quad_slices.cols['name s_start s_end parent_name k1l'].show()
print('Sum of slice k1l before [m^-1]:', quad_slices.k1l.sum())

########################################
# Change the shared expression
########################################
env['kq'] = KQ_AFTER

# Refresh attribute columns to see the current parent strength.
table_after         = line.get_table(attr = True)
quad_slices_after   = table_after.rows[table_after.parent_name == 'quad']
quad_slices_after.cols['name s_start s_end parent_name k1l'].show()
print('Sum of slice k1l after [m^-1]:', quad_slices_after.k1l.sum())
