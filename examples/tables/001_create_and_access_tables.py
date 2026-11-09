# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
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
TITLE       = 'Illustrative BPM readings'

################################################################################
# Create a table
################################################################################
table = xt.Table(
    data        = {
        'name':  np.array(BPM_NAMES),
        's':     np.array(S_M),
        'x':     np.array(X_M),
        'y':     np.array(Y_M),
        'title': TITLE},
    col_names   = ['name', 's', 'x', 'y'],
    index       = 'name')

################################################################################
# Inspect rows and columns
################################################################################

########################################
# Display the table
########################################
# col_names identifies row arrays; title is a scalar table attribute.
table.show()
print('Number of rows:', len(table))
print('Dataset:', table.title)

########################################
# Read an array or one named value
########################################
print('Horizontal readings [m]:', table.x)
print('The same column by key [m]:', table['x'])
print('Horizontal reading at bpm.2 [m]:', table['x', 'bpm.2'])

################################################################################
# Exchange columns with pandas
################################################################################

########################################
# Create a DataFrame
########################################
# The name column remains a column, not a pandas index.
dataframe = table.to_pandas()
print(dataframe)

########################################
# Return to an Xtrack table
########################################
# This exchanges columns; arbitrary scalar table attributes are not carried.
table_from_pandas = xt.Table.from_pandas(dataframe, index = 'name')
table_from_pandas.show()
