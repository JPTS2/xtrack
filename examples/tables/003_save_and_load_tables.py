# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Measurements
########################################
# Synthetic BPM readings in metres; no optics calculation is assumed.
BPM_NAMES   = ['bpm.1', 'bpm.2', 'bpm.3']
X_M         = [0.2e-3, -0.3e-3, 0.1e-3]
TITLE       = 'Illustrative BPM readings'

########################################
# Output files
########################################
# Paths relative to this script, separate from any input data.
JSON_FILE   = 'outputs/bpm_readings.json'
CSV_FILE    = 'outputs/bpm_readings.csv'

################################################################################
# Create columns and scalar metadata
################################################################################
# col_names separates row arrays from scalar attributes.
table = xt.Table(
    data        = {
        'name':  np.array(BPM_NAMES),
        'x':     np.array(X_M),
        'title': TITLE},
    col_names   = ['name', 'x'],
    index       = 'name')
table.show()
print('Dataset:', table.title)

################################################################################
# Save and reconstruct a table
################################################################################

########################################
# Reconstruct from a dictionary
########################################
payload = table.to_dict()
table_dict = xt.Table.from_dict(payload)
table_dict.show()
print('Restored dictionary title:', table_dict.title)
print('Largest dictionary x difference [m]:',
      np.max(np.abs(table_dict.x - table.x)))

########################################
# Save and reload JSON
########################################
json_path = Path(__file__).resolve().parent / JSON_FILE
json_path.parent.mkdir(parents = True, exist_ok = True)

table.to_json(json_path)
table_json = xt.Table.from_json(json_path)
table_json.show()
print('Restored JSON title:', table_json.title)
print('Largest JSON x difference [m]:',
      np.max(np.abs(table_json.x - table.x)))
print('Saved JSON:', json_path)

########################################
# Save and reload CSV
########################################
csv_path = Path(__file__).resolve().parent / CSV_FILE
csv_path.parent.mkdir(parents = True, exist_ok = True)

# Xtrack CSV includes comment metadata as well as the column data.
table.to_csv(csv_path)
table_csv = xt.Table.from_csv(csv_path)
table_csv.show()
print('Restored CSV title:', table_csv.title)
print('Largest CSV x difference [m]:',
      np.max(np.abs(table_csv.x - table.x)))

print('Saved CSV:', csv_path)
