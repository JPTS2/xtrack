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
# Native ring and analysis emittances
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
NEMITT_X        = 1e-6                    # Normalized rms emittance [m rad]
NEMITT_Y        = 2e-6                    # [m rad]

########################################
# Output file
########################################
JSON_FILE       = 'outputs/twiss.json'    # Relative to this script

################################################################################
# Calculate the optics result to preserve
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
tw          = line.twiss4d()

################################################################################
# Save and reload the Twiss result
################################################################################

########################################
# Round-trip through a dictionary
########################################
payload     = tw.to_dict()
from_dict   = xt.TwissTable.from_dict(payload)

########################################
# Write JSON and load the recorded table type
########################################
json_path = Path(__file__).resolve().parent / JSON_FILE
json_path.parent.mkdir(parents = True, exist_ok = True)
tw.to_json(json_path)
restored = xt.load(json_path)
print('Saved optics to:', json_path)
print('Loaded type:', type(restored).__name__)

################################################################################
# Inspect preserved optics and reference data
################################################################################

########################################
# Compare values and phase information
########################################
for label, optics in [
    ('Original', tw), ('Dictionary', from_dict), ('JSON', restored),
]:
    print(label, 'tunes:', optics.qx, optics.qy)
    optics.rows[['ip.1', 'bpm.s4.d']].cols[
        'name s betx bety dx mux muy'].show(maxwidth = 'full')
print('Loaded reference p0*c [eV]:', restored.particle_on_co.p0c[0])
print('Loaded W-matrix shape:', restored.W_matrix.shape)

########################################
# Continue analysis using the saved result
########################################
# W_matrix and the closed-orbit particle retain the normal-mode and reference
# information needed for covariance analysis without running Twiss again.
covariance = restored.get_beam_covariance(
    nemitt_x    = NEMITT_X,
    nemitt_y    = NEMITT_Y)
covariance.rows[['ip.1', 'bpm.s4.d']].cols[
    'name sigma_x sigma_y'].show(maxwidth = 'full')
print('Largest beta_x difference after JSON reload [m]:',
      np.max(np.abs(restored.betx - tw.betx)))

# This is a result snapshot. Save the associated model separately to perform
# tracking, matching or a new optics calculation with changed strengths.
