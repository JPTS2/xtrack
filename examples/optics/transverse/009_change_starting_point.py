# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xtrack as xt

################################################################################
# User parameters
################################################################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'
NEW_START       = 'obs.4'

################################################################################
# Calculate the original periodic solution
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
original    = line.twiss4d()

################################################################################
# Start the calculation at another entrance
################################################################################

########################################
# Cover one full turn from the new start
########################################
# start without end requests a complete wrapped turn. No line editing is needed.
wrapped = line.twiss4d(start = NEW_START)

########################################
# Inspect row order and the phase origins
########################################
for label, optics in [
    ('Original', original), ('New calculation start', wrapped),
]:
    print(label)
    optics.rows[r'obs\..*|_end_point'].cols[
        'name s mux muy betx bety'].show(maxwidth = 'full')
print('The line still starts at:', line.element_names[0])

################################################################################
# Move the coordinate zeros while keeping the original row order
################################################################################

########################################
# Set s and phase advances to zero at a selected entrance
########################################
zeroed  = line.twiss4d(zero_at = NEW_START)

########################################
# Inspect negative coordinates upstream of the chosen zero
########################################
# This does not reorder rows or wrap negative s/phase values into one turn.
zeroed.rows[r'obs\..*|_end_point'].cols[
    'name s mux muy betx bety'].show(maxwidth = 'full')

################################################################################
# Change the line order when tracking should also start there
################################################################################

########################################
# Cycle a copy of the line
########################################
# Cycling changes the element order. Keep the original line for comparison.
cycled      = line.copy()
cycled.cycle(name_first_element = NEW_START)
tw_cycled      = cycled.twiss4d()

########################################
# Compare the two full-turn solutions at named markers
########################################
print('Cycled line starts at:', cycled.element_names[0])
tw_cycled.rows[r'obs\..*|_end_point'].cols[
    'name s mux muy betx bety'].show(maxwidth = 'full')
print('Full-turn phase advances from calculation start:',
      wrapped.mux[-1] - wrapped.mux[0], wrapped.muy[-1] - wrapped.muy[0])
print('Tunes from cycled line:', tw_cycled.qx, tw_cycled.qy)
