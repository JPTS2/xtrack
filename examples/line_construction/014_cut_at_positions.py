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
print('Uncut line\n----------')
print('Positions are measured from the line entrance [m].')
table_before = line.get_table()
table_before.cols['name s_start s_end element_type'].show()
print(f'Original length [m]: {line.get_length():.3f}')

################################################################################
# Cut at the requested positions
################################################################################

########################################
# Create thick slices
########################################
# This changes the line in place. Cuts inside an element create slices;
# already existing boundaries need no extra cut.
print('\nCut at specified longitudinal positions'
      '\n---------------------------------------')
print('Cut positions [m]:', CUT_POSITIONS)
print('Cuts create thick body slices linked to their original elements.')
print('The line is edited in place; its total length is preserved.')
line.cut_at_s(CUT_POSITIONS)
table = line.get_table(attr = True)
table.cols[
    'name s_start s_end element_type parent_name'].show(maxwidth = 'full')
print('Entry/exit maps and markers have zero length; '
      'body slices fill the magnet.')
print(f'Length after cutting [m]: {line.get_length():.3f}')

########################################
# Inspect the requested boundaries
########################################
print('\nCheck the requested boundaries\n------------------------------')
print('Requested cut [m]   Nearest boundary [m]   Difference [m]')
for position in CUT_POSITIONS:
    nearest = min(table.s_start, key = lambda s: abs(s - position))
    print(f'{position:17.3f} {nearest:22.3f} {nearest - position:16.3e}')

################################################################################
# Change the parent quadrupole strength
################################################################################

########################################
# Inspect the original slice strengths
########################################
# Thick slices retain their parent element; they are not independent magnets.
print('\nQuadrupole slices retain their parent strength'
      '\n----------------------------------------------')
print('Units: s_start, s_end [m]; k1l [m^-1].')
print(f'Parent quad.k1 [m^-2]: {env["quad"].k1:.3f}')
quad_slices = table.rows[table.parent_name == 'quad']
quad_slices.cols[
    'name s_start s_end parent_name k1l'].show(maxwidth = 'full')
print(f'Sum of slice k1l [m^-1]: {quad_slices.k1l.sum():.3f}')

########################################
# Change the shared expression
########################################
env['kq'] = KQ_AFTER

# Refresh attribute columns to see the current parent strength.
table_after         = line.get_table(attr = True)
quad_slices_after   = table_after.rows[table_after.parent_name == 'quad']
print('\nChange the parent variable; slices follow'
      '\n-----------------------------------------')
print(f'Parent quad.k1 [m^-2]: {env["quad"].k1:.3f}')
print('The slices remain linked; their positions and lengths are unchanged.')
quad_slices_after.cols[
    'name s_start s_end parent_name k1l'].show(maxwidth = 'full')
print(f'Sum of slice k1l [m^-1]: {quad_slices_after.k1l.sum():.3f}')
