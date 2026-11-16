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

########################################
# Model parameters
########################################
QUAD_LENGTH = 0.4                       # [m]
KQ_START    = 0.2                       # [m^-2]
KQ_AFTER    = 0.3                       # [m^-2]
P0C_START   = 1.0e9                     # Proton momentum times c [eV]
P0C_AFTER   = 2.0e9                     # [eV]
N_SLICES    = 2

########################################
# Output file
########################################
# Relative to this script, separate from any input models.
SAVE_FILE   = 'outputs/shared_environment.json'

################################################################################
# Build lines sharing elements and expressions
################################################################################

########################################
# Define common elements
########################################
env = xt.Environment()
env['kq'] = KQ_START
env['p0c'] = P0C_START
env.new_particle('reference', 'proton', p0c = 'p0c')
env.particle_ref = 'reference'
env.new('qf', xt.Quadrupole, length = QUAD_LENGTH, k1 = 'kq')
env.new('qd', xt.Quadrupole, length = QUAD_LENGTH, k1 = '-kq')
env.new('drift', xt.Drift, length = 1.0)

########################################
# Register both lines
########################################
line    = env.new_line(
    name        = 'transport',
    components  = ['qf', 'drift', 'qd'],
)
env.new_line(
    name        = 'doublet',
    components  = ['qf', 'qd'],
)

########################################
# Include a sliced line with shared parents
########################################
sliced = env.new_line(name = 'sliced', components = line.element_names)
print('Save one environment with three related lines'
      '\n---------------------------------------------')
print('transport and doublet share magnets; sliced refers to their parents.')
sliced.slice_thick_elements(
    slicing_strategies  = [
        xt.Strategy(
            slicing         = xt.Teapot(N_SLICES),
            element_type    = xt.Quadrupole,
        ),
    ],
)

################################################################################
# Save the environment and reload it independently
################################################################################
save_path = Path(__file__).resolve().parent / SAVE_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)

# Save all three lines, their shared parent magnets and the named reference.
env.to_json(save_path)
print('Saved environment:', save_path)

################################################################################
# Check what reload preserves before changing any variables
################################################################################

########################################
# Compare geometry and shared definitions
########################################
print('\nReload preserves values and sharing'
      '\n-----------------------------------')
env_loaded      = xt.load(save_path)
line_loaded     = env_loaded['transport']
doublet_loaded  = env_loaded['doublet']
print(f'{"Line":<12} {"Original [m]":>14} {"Reloaded [m]":>14}')
for name in ['transport', 'doublet', 'sliced']:
    print(f'{name:<12} {env[name].get_length():14.3f}'
          f' {env_loaded[name].get_length():14.3f}')
print('Reloaded transport and doublet share qf:',
      line_loaded.get('qf') is doublet_loaded.get('qf'))

########################################
# Check the restored strength value and expression
########################################
print(f'qf.k1 [m^-2]: original = {line["qf"].k1:.3f}, '
      f'reloaded = {line_loaded["qf"].k1:.3f}')
print('Restored qf.k1 expression:', env_loaded.ref['qf'].k1.xdeps.expr)

################################################################################
# Change a knob only in the loaded environment
################################################################################

########################################
# Show that both loaded lines still share one live magnet
########################################
print('\nOne loaded knob updates both lines and their slices'
      '\n---------------------------------------------------')
env_loaded['kq'] = KQ_AFTER
print(f'Loaded kq [m^-2]: {env_loaded["kq"]:.3f}')
print(f'Loaded transport qf.k1 [m^-2]: {line_loaded["qf"].k1:.3f}')
print(f'Loaded doublet qf.k1 [m^-2]: {doublet_loaded["qf"].k1:.3f}')
print(f'Original qf.k1 [m^-2]: {line["qf"].k1:.3f} (unchanged)')

########################################
# Compare magnetic slices through their original and loaded parents
########################################
# Show the magnetic kicks only; boundary maps and drifts are not the lesson.
for label, current_line in [
    ('Original magnetic slices', sliced),
    ('Modified loaded magnetic slices', env_loaded['sliced']),
]:
    print('\n' + label + '; k1l [m^-1]:')
    table = current_line.get_table(attr = True)
    table.rows[table.element_type == 'ThinSliceQuadrupole'].cols[
        'name parent_name k1l'].show()

################################################################################
# Change the named reference through its restored variable
################################################################################
print('\nThe named reference remains live after reload'
      '\n---------------------------------------------')
print(f'Loaded momentum before changing p0c [GeV/c]: '
      f'{line_loaded.particle_ref.p0c[0] * 1e-9:.3f}')
env_loaded['p0c'] = P0C_AFTER
print('After changing p0c only in the loaded environment:')
for name in ['transport', 'doublet', 'sliced']:
    print(f'Loaded {name} momentum [GeV/c]: '
          f'{env_loaded[name].particle_ref.p0c[0] * 1e-9:.3f}')
print(f'Original momentum [GeV/c]: '
      f'{line.particle_ref.p0c[0] * 1e-9:.3f} (unchanged)')
