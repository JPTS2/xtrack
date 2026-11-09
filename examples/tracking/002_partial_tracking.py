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
N_ELEMENTS      = 10
DRIFT_LENGTH    = 1.0                      # [m]
P0C             = 7.0e12                    # Proton reference p0*c [eV]
X               = 1.0e-3                    # [m]
PX              = 1.0e-5                    # Normalized canonical momentum

################################################################################
# Build a line with transparent element indexing
################################################################################
line = xt.Line(
    elements        = [
        xt.Drift(length = DRIFT_LENGTH) for _ in range(N_ELEMENTS)
    ],
    element_names   = [f'drift.{ii}' for ii in range(N_ELEMENTS)],
)
line.set_particle_ref('proton', p0c = P0C)

# Keep s cumulative here so that traversed length is visible across turns.
line.reset_s_at_end_turn = False
initial = line.build_particles(x = X, px = PX)

################################################################################
# Stop before an element and resume
################################################################################

########################################
# Track through elements 0 to 4
########################################
# ele_start is inclusive; ele_stop is exclusive. No end-of-line crossing yet.
partial = initial.copy()
line.track(partial, ele_stop = 'drift.5')
partial.get_table().cols['x s at_turn at_element'].show(maxwidth = 'full')

########################################
# Continue from the stored boundary
########################################
# Keep ele_start consistent with the particle's at_element and s.
line.track(partial, ele_start = 'drift.5')
partial.get_table().cols['x s at_turn at_element'].show(maxwidth = 'full')

################################################################################
# Cross the end of the line
################################################################################

########################################
# Prepare particles at element 7
########################################
wrapped = initial.copy()
line.track(wrapped, ele_stop = 'drift.7')

########################################
# Stop at element 3 of the next turn
########################################
# Traverses 7, 8, 9, 0, 1, 2: one boundary crossing increments at_turn.
line.track(wrapped, ele_start = 'drift.7', ele_stop = 'drift.3')
wrapped.get_table().cols['x s at_turn at_element'].show(maxwidth = 'full')

################################################################################
# Count elements or extend a partial path over turns
################################################################################

########################################
# Specify the number of elements
########################################
counted = initial.copy()
line.track(counted, num_elements = N_ELEMENTS + 3)
counted.get_table().cols['x s at_turn at_element'].show(maxwidth = 'full')

########################################
# Use num_turns with a stop inside the line
########################################
# With start = 0 and stop downstream, num_turns = 3 gives two complete turns
# followed by the final partial turn. at_turn counts line-end crossings.
multiple = initial.copy()
line.track(multiple, ele_stop = 'drift.3', num_turns = 3)
multiple.get_table().cols['x s at_turn at_element'].show(maxwidth = 'full')

########################################
# Extend a path that wraps around the line end
########################################
# With stop upstream of start, the first path already crosses the line end.
# num_turns = 3 then gives three crossings before reaching element 3.
wrapped_multiple = initial.copy()
line.track(wrapped_multiple, ele_stop = 'drift.7')
line.track(
    wrapped_multiple,
    ele_start   = 'drift.7',
    ele_stop    = 'drift.3',
    num_turns   = 3,
)
wrapped_multiple.get_table().cols[
    'x s at_turn at_element'].show(maxwidth = 'full')
