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
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAME       = 'ring'

########################################
# Saved model
########################################
# Saved file relative to this example.
SAVE_FILE       = 'outputs/getting_started_ring.json'

########################################
# Quadrupole strength
########################################
KQF_SCALE       = 1.01                  # Scale kqf in the loaded copy

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
env.to_json(save_path)
print(f'Saved environment: {save_path}')

################################################################################
# Reload the environment
################################################################################
env_loaded  = xt.load(save_path)
line_loaded = env_loaded[LINE_NAME]
tw_loaded   = line_loaded.twiss(method = '4d')

print(f'Original tunes: Qx = {tw0.qx:.6f}, '
      f'Qy = {tw0.qy:.6f}')
print(f'Reloaded tunes: Qx = {tw_loaded.qx:.6f}, '
      f'Qy = {tw_loaded.qy:.6f}')

################################################################################
# Change a knob in the loaded copy
################################################################################
# The restored expressions still link every qf instance to kqf. The original
# environment is independent and keeps its starting strengths.
env_loaded['kqf'] *= KQF_SCALE
print(f'Original qf.1 k1: {line["qf.1"].k1:.9f} m^-2')
print(f'Loaded qf.1 k1:   {line_loaded["qf.1"].k1:.9f} m^-2')

tw = line_loaded.twiss(method = '4d')
print(f'Changed-copy tunes: Qx = {tw.qx:.6f}, '
      f'Qy = {tw.qy:.6f}')
