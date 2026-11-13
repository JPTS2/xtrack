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
# Output files
########################################
JSON_FILE       = 'outputs/twiss.json'    # Relative to this script
CSV_FILE        = 'outputs/twiss.csv'
HDF5_FILE       = 'outputs/twiss.h5'
TFS_LAYOUT      = 'outputs/twiss_madx_layout.tfs'
TFS_CONVENTIONS = 'outputs/twiss_madx_conventions.tfs'

################################################################################
# Calculate the optics result to preserve
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
tw          = line.twiss4d(strengths = True)

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

########################################
# Preserve the same metadata in native CSV and HDF5
########################################
# These are Xtrack result formats, including reference and normal-mode data.
# A plain column-only CSV from another application has no such guarantee.
csv_path    = Path(__file__).resolve().parent / CSV_FILE
hdf5_path   = Path(__file__).resolve().parent / HDF5_FILE
tw.to_csv(csv_path)
tw.to_hdf5(hdf5_path)
from_csv    = xt.load(csv_path)
from_hdf5   = xt.load(hdf5_path)

################################################################################
# Inspect preserved optics and reference data
################################################################################

########################################
# Compare values and phase information
########################################
for label, optics in [
    ('Original', tw), ('Dictionary', from_dict), ('JSON', restored),
    ('CSV', from_csv), ('HDF5', from_hdf5),
]:
    print(label, 'tunes:', optics.qx, optics.qy)
    optics.rows[['ip.1', 'bpm.s4.d']].cols[
        'name s betx bety dx mux muy'].show(maxwidth = 'full')
    print('Reference p0*c [eV]:', optics.particle_on_co.p0c[0])
    print('W-matrix shape:', optics.W_matrix.shape)

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

################################################################################
# Separate a text layout from a change of optics conventions
################################################################################

########################################
# Write MAD-X field names with Xtrack entry-point values
########################################
# The preset only selects/formats fields. It retains the synthetic endpoint
# and Xtrack's delta derivatives; this alone is not MAD-X TWISS conversion.
layout_path = Path(__file__).resolve().parent / TFS_LAYOUT
tw.to_tfs(layout_path, format_spec = xt.format_spec.madx_twiss_v1)

########################################
# Prepare complete forward on-momentum periodic 4D exit optics
########################################
# Do this before row selection. Exit optics come from the next entry row;
# names and strengths still belong to the element traversed in that row.
spec    = xt.format_spec.madx_twiss_v1
columns = [field['source_name'] for field in spec['columns']]
data    = {name: tw[name][:-1].copy() for name in columns}
for name in ['s', 'x', 'y', 'px', 'py', 'betx', 'bety', 'alfx', 'alfy',
             'mux', 'muy', 'dx', 'dy', 'dpx', 'dpy']:
    data[name] = tw[name][1:].copy()

# At delta=0, d(ptau)/d(delta)=beta0. MAD-X normalizes these derivatives to
# ptau; Xtrack normalizes them to delta. This applies to chromaticities too.
for name in ['dx', 'dy', 'dpx', 'dpy']:
    data[name] /= tw.beta0
for field in spec['scalars']:
    if 'source_name' in field:
        name        = field['source_name']
        data[name]  = tw[name]
data['dqx'] /= tw.beta0
data['dqy'] /= tw.beta0
exit_optics         = xt.Table(data = data, col_names = columns)
conventions_path    = Path(__file__).resolve().parent / TFS_CONVENTIONS
exit_optics.to_tfs(conventions_path, format_spec = spec)

########################################
# Inspect the selected text data rather than a complete snapshot
########################################
# Strengths keep Xtrack's definitions, including implicit design bend fields.
# Neither text layout includes particle_on_co or W_matrix for later analysis.
text_table = xt.Table.from_tfs(conventions_path)
text_table.rows[:8].cols['name s betx dx'].show()
print('Entrance/exit rows:', len(tw.name), len(text_table.name))
print('Native / exported dQx:', tw.dqx, text_table.dq1)
print('Saved layout and converted optics:', layout_path, conventions_path)

# This is a result snapshot. Save the associated model separately to perform
# tracking, matching or a new optics calculation with changed strengths.
