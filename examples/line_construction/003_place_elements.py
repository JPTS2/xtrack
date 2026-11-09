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
QUAD_LENGTH = 0.4                       # [m]
QF_START    = 1.0                       # Entrance position [m]
QD_GAP      = 1.2                       # Gap from qf exit to qd entrance [m]
LINE_LENGTH = 5.0                       # [m]

################################################################################
# Place the elements
################################################################################

########################################
# Define the elements
########################################
env = xt.Environment()
env.new('qf', xt.Quadrupole, length = QUAD_LENGTH, k1 = 0.2)
env.new('qd', xt.Quadrupole, length = QUAD_LENGTH, k1 = -0.2)
env.new('end', xt.Marker)

########################################
# Assemble the positioned line
########################################
# refer sets the default anchor of the placed object. from_ identifies the
# anchor on the reference object. Explicit anchors avoid centre/edge ambiguity.
line = env.new_line(
    refer       = 'start',
    length      = LINE_LENGTH,
    components  = [
        env.place('qf', at = QF_START),
        env.place('qd', at = QD_GAP, from_ = 'qf@end'),
        env.place('end', at = LINE_LENGTH),
    ],
)

################################################################################
# Inspect the positions
################################################################################
table = line.get_table()
table.cols['name s_start s_center s_end element_type'].show()
print(f'Gap between quadrupoles: '
      f'{table["s_start", "qd"] - table["s_end", "qf"]:.3f} m')

################################################################################
# Compare element anchors
################################################################################

########################################
# Place centres and ends
########################################
# The default refer is 'center'. Position qf's centre to recover its original
# entrance, and position qd's end relative to qf's end. from_anchor names the
# reference object's anchor; anchor names the object being placed.
line_anchors = env.new_line(
    length      = LINE_LENGTH,
    components  = [
        env.place('qf', at = QF_START + QUAD_LENGTH / 2),
        env.place(
            'qd',
            anchor      = 'end',
            at          = QD_GAP + QUAD_LENGTH,
            from_       = 'qf',
            from_anchor = 'end',
        ),
        env.place('end', at = LINE_LENGTH),
    ],
)

########################################
# Compare the equivalent positions
########################################
print('Original start-anchor placements:')
table.cols['name s_start s_center s_end'].show()
print('Default centre and explicit end-anchor placements:')
table_anchors = line_anchors.get_table()
table_anchors.cols['name s_start s_center s_end'].show()
