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

########################################
# Output file
########################################
# Relative to this script, separate from any input models.
SAVE_FILE   = 'outputs/shared_environment.json'

################################################################################
# Build two lines sharing elements and expressions
################################################################################

########################################
# Define common elements
########################################
env = xt.Environment()
env['kq'] = KQ_START
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
doublet = env.new_line(
    name        = 'doublet',
    components  = ['qf', 'qd'],
)
print('Original line names:', list(env.lines.keys()))

################################################################################
# Save and restore the complete environment
################################################################################

########################################
# Save as native JSON
########################################
save_path = Path(__file__).resolve().parent / SAVE_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)

# Save the environment so both lines and their common definitions are included.
env.to_json(save_path)
print('Saved environment:', save_path)

########################################
# Reload and inspect the lines
########################################
env_loaded      = xt.load(save_path)
line_loaded     = env_loaded['transport']
doublet_loaded  = env_loaded['doublet']
print('Reloaded line names:', list(env_loaded.lines.keys()))
print('Original and restored transport lengths [m]:',
      line.get_length(), line_loaded.get_length())
print('Original and restored doublet lengths [m]:',
      doublet.get_length(), doublet_loaded.get_length())

################################################################################
# Use the restored dependencies
################################################################################

########################################
# Inspect the restored expressions
########################################
env_loaded.info('qf')
print('Reloaded transport and doublet k1 before [m^-2]:',
      line_loaded['qf'].k1, doublet_loaded['qf'].k1)

########################################
# Change only the loaded environment
########################################
env_loaded['kq'] = KQ_AFTER
print('Reloaded transport and doublet k1 after [m^-2]:',
      line_loaded['qf'].k1, doublet_loaded['qf'].k1)
print('Reloaded qd k1 after [m^-2]:', line_loaded['qd'].k1)
print('Original transport and doublet k1 [m^-2]:',
      line['qf'].k1, doublet['qf'].k1)
