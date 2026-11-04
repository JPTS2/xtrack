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
CELL_LENGTH = 3.0                       # [m]
CELL_START  = 2.0                       # [m]

################################################################################
# Build a reusable subline
################################################################################
env = xt.Environment()
cell = env.new_line(
    name        = 'cell',
    length      = CELL_LENGTH,
    refer       = 'start',
    components  = [
        env.new(
            'qf', xt.Quadrupole,
            length  = QUAD_LENGTH,
            k1      = 0.2,
            at      = 0.0,
        ),
        env.new(
            'qd', xt.Quadrupole,
            length  = QUAD_LENGTH,
            k1      = -0.2,
            at      = CELL_LENGTH - QUAD_LENGTH,
        ),
    ],
)

################################################################################
# Compare subline anchors
################################################################################

########################################
# Place equivalent cell intervals
########################################
# These anchors locate the entire subline, not its first element's centre.
# With these defaults, all three placements give the interval [2 m, 5 m].
line_start = env.new_line(
    components  = [
        env.place('cell', anchor = 'start', at = CELL_START),
    ],
)
line_center = env.new_line(
    components  = [
        env.place('cell', anchor = 'center', at = CELL_START + CELL_LENGTH / 2),
    ],
)
line_end = env.new_line(
    components  = [
        env.place('cell', anchor = 'end', at = CELL_START + CELL_LENGTH),
    ],
)

########################################
# Compare the element positions
########################################
for label, line in [('Start anchor', line_start),
                    ('Centre anchor', line_center),
                    ('End anchor', line_end)]:
    print(label)
    line.get_table().cols['name s_start s_end'].show()

################################################################################
# Place another subline relative to the first
################################################################################

########################################
# Give the sublines distinct element names
########################################
# replicate gives each occurrence a distinct name while referencing the same
# original magnets. See 006_clones_and_replicas.py for the ownership comparison.
# Here qd ends at the cell exit; a cell with a trailing drift would differ.
env.new_line(
    name        = 'first_cell',
    components  = [cell.replicate(suffix = 'first')],
)
env.new_line(
    name        = 'second_cell',
    components  = [cell.replicate(suffix = 'second')],
)

########################################
# Place the second cell at the first cell exit
########################################
line = env.new_line(
    components  = [
        env.place('first_cell', anchor = 'start', at = CELL_START),
        env.place('second_cell', anchor = 'start', at = 'qd.first@end'),
    ],
)

########################################
# Inspect the combined line
########################################
line.get_table().cols['name s_start s_end isreplica'].show()
