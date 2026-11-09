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
# HDF5 needs h5py; TFS reading needs tfs-pandas. No external engine runs.
HDF5_FILE   = 'outputs/bpm_readings.h5'
HDF5_GROUP  = 'measurements'
TFS_FILE    = 'outputs/bpm_readings.tfs'

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
# Save and reload HDF5
################################################################################
hdf5_path = Path(__file__).resolve().parent / HDF5_FILE
hdf5_path.parent.mkdir(parents = True, exist_ok = True)

table.to_hdf5(hdf5_path, group = HDF5_GROUP)
table_hdf5 = xt.Table.from_hdf5(hdf5_path, group = HDF5_GROUP)
table_hdf5.show()
print('Restored HDF5 title:', table_hdf5.title)
print('Largest HDF5 x difference [m]:',
      np.max(np.abs(table_hdf5.x - table.x)))

################################################################################
# Write a defined TFS layout
################################################################################

########################################
# Choose columns, headers and precision
########################################
# The specification selects the output fields, not a new beam model.
format_spec = {
    'version':      '1',
    'float_format': '.16e',
    'columns': [
        {'name': 'name', 'type': 'string'},
        {'name': 'x',    'type': 'float'}],
    'scalars': [
        {'name': 'title', 'type': 'string'}]}

########################################
# Write and read the text file
########################################
tfs_path = Path(__file__).resolve().parent / TFS_FILE
tfs_path.parent.mkdir(parents = True, exist_ok = True)

table.to_tfs(tfs_path, format_spec = format_spec)
table_tfs = xt.Table.from_tfs(tfs_path)
table_tfs.show()
print('Restored TFS title:', table_tfs.title)
print('Largest TFS x difference [m]:',
      np.max(np.abs(table_tfs.x - table.x)))

print('Saved HDF5:', hdf5_path)
print('Saved TFS:', tfs_path)
