# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import json
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
# Reference for the element-table export
########################################
P0C         = 1e9                        # Electron reference momentum [eV/c]
DRIFT_LEN   = 1.0                        # [m]
QUAD_LEN    = 0.25                       # [m]
K1          = 0.8                        # [m^-2]

########################################
# Output files
########################################
# TFS reading needs tfs-pandas. No external engine runs.
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
        {'name': 'NAME',    'source_name': 'name', 'type': 'string'},
        {'name': 'ORBIT_X', 'source_name': 'x',    'type': 'float',
         'format': '.9f'}],
    'scalars': [
        {'name': 'DATASET', 'source_name': 'title', 'type': 'string'}]}

########################################
# Write and read the text file
########################################
tfs_path = Path(__file__).resolve().parent / TFS_FILE
tfs_path.parent.mkdir(parents = True, exist_ok = True)

table.to_tfs(tfs_path, format_spec = format_spec)
table_tfs = xt.Table.from_tfs(tfs_path)
table_tfs.show()
print('Restored TFS dataset:', table_tfs.dataset)
print('Largest TFS x difference [m]:',
      np.max(np.abs(table_tfs.orbit_x - table.x)))

########################################
# Load the same layout from JSON or TOML
########################################
json_path = tfs_path.with_name('bpm_format.json')
json_path.write_text(json.dumps(format_spec, indent = 2), encoding = 'utf-8')
toml_path = Path(__file__).with_name('bpm_format.toml')
for label, source in [('json', json_path), ('toml', toml_path)]:
    output = tfs_path.with_name(f'bpm_from_{label}.tfs')
    table.to_tfs(output, format_spec = source)
    print(label, 'layout gives identical text:',
          output.read_text() == tfs_path.read_text())

################################################################################
# Tailor a built-in layout to the available columns
################################################################################

########################################
# Make an independent copy of the preset
########################################
# The full preset needs a Twiss table. Keep only its name and x fields here;
# measurements do not acquire optics columns simply by choosing a preset.
measurement_spec = xt.format_spec.load(xt.format_spec.optics_orbit_v1)
measurement_spec['columns'] = [
    field for field in measurement_spec['columns']
    if field['source_name'] in ['name', 'x']]
measurement_spec['scalars'] = format_spec['scalars']
measurement_spec['float_format'] = '.12e'

########################################
# Write the modified layout
########################################
preset_path = tfs_path.with_name('bpm_from_preset.tfs')
table.to_tfs(preset_path, format_spec = measurement_spec)
print(preset_path.read_text())
print('Unchanged built-in columns:',
      [field['name'] for field in xt.format_spec.optics_orbit_v1['columns']])

################################################################################
# Export an element table with physical-reference headers
################################################################################

########################################
# Derive a column and collect scalar metadata
########################################
# This is a short transport line, not an optics calculation.
line = xt.Line(
    elements        = [xt.Drift(length = DRIFT_LEN),
                       xt.Quadrupole(length = QUAD_LEN, k1 = K1)],
    element_names   = ['drift', 'quad'])
line.particle_ref = xt.Particles('electron', p0c = P0C)
elements = line.get_table(attr = True).rows[:-1]
reference = line.particle_ref
keywords = np.array([kind.upper() for kind in elements.element_type])
export_table = xt.Table(
    data        = {
        'name':       elements.name,
        'keyword':    keywords,
        'length':     elements.length,
        'mass_GeV':   reference.mass0 * 1e-9,
        'charge_e':   reference.q0,
        'energy_GeV': reference.energy0[0] * 1e-9,
        'pc_GeV':     reference.p0c[0] * 1e-9},
    col_names   = ['name', 'keyword', 'length'],
    index       = 'name')

########################################
# Map internal names to the output convention
########################################
# Format specifications rename fields; unit conversions belong in the data.
element_spec = {
    'version':      '1',
    'float_format': '.12e',
    'columns': [
        {'name': 'NAME',    'source_name': 'name',    'type': 'string'},
        {'name': 'KEYWORD', 'source_name': 'keyword', 'type': 'string'},
        {'name': 'L',       'source_name': 'length',  'type': 'float'}],
    'scalars': [
        {'name': 'MASS',   'source_name': 'mass_GeV',   'type': 'float'},
        {'name': 'CHARGE', 'source_name': 'charge_e',   'type': 'float'},
        {'name': 'ENERGY', 'source_name': 'energy_GeV', 'type': 'float'},
        {'name': 'PC',     'source_name': 'pc_GeV',     'type': 'float'}]}
element_path = tfs_path.with_name('element_description.tfs')
export_table.to_tfs(element_path, format_spec = element_spec)
element_table = xt.Table.from_tfs(element_path)
element_table.show()
print('Reloaded mass [GeV/c^2]:', element_table.mass)
print('Reloaded charge [e]:', element_table.charge)
print('Reloaded energy [GeV]:', element_table.energy)
print('Reloaded momentum [GeV/c]:', element_table.pc)
print('Outputs:', tfs_path.parent)
