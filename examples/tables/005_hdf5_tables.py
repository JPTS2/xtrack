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
BPM_NAMES   = ['bpm.1', 'bpm.2', 'bpm.3']
X_M         = [0.2e-3, -0.3e-3, 0.1e-3]   # Synthetic readings [m]
TITLE       = 'Illustrative BPM readings'
HDF5_FILE   = 'outputs/bpm_readings.h5'    # Relative to this script
HDF5_GROUP  = 'measurements'

################################################################################
# Create columns and scalar metadata
################################################################################
# col_names separates row arrays from scalar attributes. HDF5 needs h5py.
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
