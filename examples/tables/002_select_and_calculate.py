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
print('Select rows by name\n-------------------')
print('s, x, y [m]')
print('Named rows: bpm.1 and bpm.3')
table.rows[['bpm.1', 'bpm.3']].show()
print('\nRegular expression: bpm[.][23]')
table.rows['bpm[.][23]'].show()

########################################
# Ranges and conditions
########################################
# Named ranges include both endpoints; integer slices exclude the stop.
print('\nSelect by range or condition\n----------------------------')
print('Named range: bpm.2 through bpm.4, including both endpoints')
table.rows['bpm.2':'bpm.4'].show()
print('\nInteger slice: rows[1:3], excluding row 3')
table.rows[1:3].show()

# Select by longitudinal position, or by a NumPy boolean condition.
print('\nPosition range: s from 1 to 2 m')
table.rows[1.0:2.0:'s'].show()
print('\nBoolean condition: abs(x) > 0.25 mm')
table.rows[np.abs(table.x) > 0.25e-3].show()

########################################
# Match another column
########################################
# Match region labels, then select the columns to display.
print('\nMatch labels in another column\n------------------------------')
print('region matches upstream')
table.rows.match(region = 'upstream').cols['s x'].show()
print('\nregion does not match upstream')
table.rows.match_not(region = 'upstream').cols['s x'].show()

################################################################################
# Select and calculate columns
################################################################################

########################################
# Evaluate an expression for display
########################################
# Expressions in cols create a result table; they do not add source columns.
print('\nCalculate columns\n-----------------')
print('Display a radial-offset expression [m], without storing a column')
table.cols['s x y sqrt(x**2+y**2)'].show()

########################################
# Store a derived column
########################################
# This is a stored numeric column, not a live expression linking x and y.
table['r_mm'] = np.sqrt(table.x**2 + table.y**2) * 1e3
print('\nStore r_mm [mm], then display bpm.2 and bpm.3')
table.rows['bpm[.][23]'].cols['s r_mm'].show()
print(f'Largest radial offset [mm]: {table.r_mm.max():.6f}')

################################################################################
# Edit an independent copy
################################################################################
# Copy before editing; the derived r_mm column remains a snapshot.
table_copy = deepcopy(table)
table_copy['x', 'bpm.2'] = 0.0
print('\nEdit an independent copy\n------------------------')
print('Original bpm.2 x [m]:', table['x', 'bpm.2'])
print('Edited copy bpm.2 x [m]:', table_copy['x', 'bpm.2'])

# Stored derived values do not follow subsequent edits to their inputs.
r_stored        = table_copy['r_mm', 'bpm.2']
r_recomputed    = np.hypot(
    table_copy['x', 'bpm.2'], table_copy['y', 'bpm.2']) * 1e3
print(f'Stored radius after editing x [mm]: {r_stored:.6f}')
print(f'Recomputed radius [mm]: {r_recomputed:.6f}')
