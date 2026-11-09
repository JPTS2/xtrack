# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from copy import deepcopy

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
# Synthetic BPM readings; positions and offsets are in metres.
BPM_NAMES   = ['bpm.1', 'bpm.2', 'bpm.3', 'bpm.4']
S_M         = [0.0, 1.0, 2.0, 3.0]
X_M         = [0.2e-3, -0.3e-3, 0.1e-3, -0.4e-3]
Y_M         = [0.1e-3, 0.2e-3, -0.1e-3, 0.0]
REGIONS     = ['upstream', 'upstream', 'downstream', 'downstream']

################################################################################
# Create a table
################################################################################
table = xt.Table(
    data    = {
        'name':   np.array(BPM_NAMES),
        's':      np.array(S_M),
        'x':      np.array(X_M),
        'y':      np.array(Y_M),
        'region': np.array(REGIONS)},
    index   = 'name')

################################################################################
# Select rows
################################################################################

########################################
# Names and regular expressions
########################################
table.rows[['bpm.1', 'bpm.3']].show()
table.rows['bpm[.][23]'].show()

########################################
# Ranges and conditions
########################################
# Named ranges include both endpoints; integer slices exclude the stop.
table.rows['bpm.2':'bpm.4'].show()
table.rows[1:3].show()

# Select by longitudinal position, or by a NumPy boolean condition.
table.rows[1.0:2.0:'s'].show()
table.rows[np.abs(table.x) > 0.25e-3].show()

########################################
# Match another column
########################################
# Match region labels, then select the columns to display.
table.rows.match(region = 'upstream').cols['s x'].show()
table.rows.match_not(region = 'upstream').cols['s x'].show()

################################################################################
# Select and calculate columns
################################################################################

########################################
# Evaluate an expression for display
########################################
# Expressions in cols create a result table; they do not add source columns.
table.cols['s x y sqrt(x**2+y**2)'].show()

########################################
# Store a derived column
########################################
# This is a stored numeric column, not a live expression linking x and y.
table['r_mm'] = np.sqrt(table.x**2 + table.y**2) * 1e3
table.rows['bpm[.][23]'].cols['s r_mm'].show()
print('Largest measured radial offset [mm]:', table.r_mm.max())

################################################################################
# Edit an independent copy
################################################################################
# Copy before editing; the derived r_mm column remains a snapshot.
table_copy = deepcopy(table)
table_copy['x', 'bpm.2'] = 0.0
print('Original bpm.2 x [m]:', table['x', 'bpm.2'])
print('Edited copy bpm.2 x [m]:', table_copy['x', 'bpm.2'])

# Stored derived values do not follow subsequent edits to their inputs.
r_stored        = table_copy['r_mm', 'bpm.2']
r_recomputed    = np.hypot(
    table_copy['x', 'bpm.2'], table_copy['y', 'bpm.2']) * 1e3
print('Stored radius after editing x [mm]:', r_stored)
print('Recomputed radius [mm]:', r_recomputed)
