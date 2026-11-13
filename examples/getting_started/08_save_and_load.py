# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/line_construction/
#                   015_save_and_load_environment.py

################################################################################
# Required packages
################################################################################
from pathlib import Path

import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model
########################################
# Lattice file relative to the repository root.
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'

########################################
# Saved model
########################################
# Saved file relative to this example.
SAVE_FILE       = 'outputs/getting_started_demo.json'

########################################
# Quadrupole strength
########################################
KQF_SCALE       = 1.001                  # Scale kqf in the loaded copy

################################################################################
# Load the model
################################################################################
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]
tw0       = line.twiss(method = '4d')

################################################################################
# Save the environment
################################################################################
save_path = Path(__file__).resolve().parent / SAVE_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)

# Save the environment, including its strength expressions.
print('Save the environment\n--------------------')
env.to_json(save_path)
print(f'Saved environment: {save_path}')

################################################################################
# Reload the environment
################################################################################
print('\nReload the environment\n----------------------')
env_loaded  = xt.load(save_path)
line_loaded = env_loaded[LINE_NAME]
tw_loaded   = line_loaded.twiss(method = '4d')

print(f'Original tunes: Qx = {tw0.qx:.6f}, '
      f'Qy = {tw0.qy:.6f}')
print(f'Reloaded tunes: Qx = {tw_loaded.qx:.6f}, '
      f'Qy = {tw_loaded.qy:.6f}')
print(f'Original arc.s1.c1.qf.in k1 [m^-2]: {line["arc.s1.c1.qf.in"].k1:.9f}')
print(f'Reloaded arc.s1.c1.qf.in k1 [m^-2]: '
      f'{line_loaded["arc.s1.c1.qf.in"].k1:.9f}')

################################################################################
# Change a knob in the loaded copy
################################################################################
# The restored expressions still link the arc QF magnets to kqf. The original
# environment is independent and keeps its starting strengths.
print('\nChange a strength in the loaded copy'
      '\n------------------------------------')
print(f'kqf scale factor: {KQF_SCALE:g}')
env_loaded['kqf'] *= KQF_SCALE
print(f'Unchanged original k1 [m^-2]: {line["arc.s1.c1.qf.in"].k1:.9f}')
print(f'Modified copy k1 [m^-2]: '
      f'{line_loaded["arc.s1.c1.qf.in"].k1:.9f}')

tw = line_loaded.twiss(method = '4d')
print(f'Changed-copy tunes: Qx = {tw.qx:.6f}, '
      f'Qy = {tw.qy:.6f}')
