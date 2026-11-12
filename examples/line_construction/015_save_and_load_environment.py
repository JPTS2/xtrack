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
P0C_START   = 1.0e9                     # Proton reference p0*c [eV]
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
env.new_particle('reference', 'proton', p0c = ['p0c'])
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
doublet = env.new_line(
    name        = 'doublet',
    components  = ['qf', 'qd'],
)

########################################
# Include a sliced line with shared parents
########################################
sliced = env.new_line(name = 'sliced', components = line.element_names)
sliced.slice_thick_elements(
    slicing_strategies  = [
        xt.Strategy(
            slicing         = xt.Teapot(N_SLICES),
            element_type    = xt.Quadrupole,
        ),
    ],
)
print('Original line names:', list(env.lines.keys()))
print('Original slices and their parents:')
sliced.get_table(attr = True).cols['name parent_name k1l'].show()

################################################################################
# Save and restore the complete environment
################################################################################

########################################
# Make an independent copy and save native JSON
########################################
save_path = Path(__file__).resolve().parent / SAVE_FILE
save_path.parent.mkdir(parents = True, exist_ok = True)

# Save all three lines and their common definitions.
env_copied = env.copy()
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

########################################
# Inspect the restored slice dependencies
########################################
print('Restored slices follow the changed parent strength:')
env_loaded['sliced'].get_table(attr = True).cols[
    'name parent_name k1l'].show()
print('Original slices remain unchanged:')
sliced.get_table(attr = True).cols['name parent_name k1l'].show()

################################################################################
# Preserve the named reference binding
################################################################################

########################################
# Change only the restored reference variable
########################################
env_loaded['p0c'] = P0C_AFTER
print('Restored transport and sliced reference p0*c [GeV]:',
      line_loaded.particle_ref.p0c[0] * 1e-9,
      env_loaded['sliced'].particle_ref.p0c[0] * 1e-9)

########################################
# Compare with the original and in-memory copy
########################################
print('Original and copied reference p0*c [GeV]:',
      line.particle_ref.p0c[0] * 1e-9,
      env_copied['transport'].particle_ref.p0c[0] * 1e-9)
env_copied['p0c'] = P0C_AFTER
print('Copied reference after changing its own variable [GeV]:',
      env_copied['transport'].particle_ref.p0c[0] * 1e-9)
