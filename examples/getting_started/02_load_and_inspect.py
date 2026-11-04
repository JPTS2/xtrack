# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/line_construction/

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

################################################################################
# Load the model
################################################################################
# Loading this Python lattice creates a fresh environment. No previous example
# needs to have run, and the file sets the reference particle explicitly.
repo_root = Path(__file__).resolve().parents[2]
env       = xt.load(repo_root / LATTICE_FILE)
line      = env[LINE_NAME]

################################################################################
# Inspect the element table
################################################################################

########################################
# Overall table
########################################
table = line.get_table()

# s_start and s_end are layout coordinates, in metres.
table.rows[:12].cols['name s_start s_end element_type'].show()

########################################
# Quadrupoles only
########################################
quadrupoles = table.rows[table.element_type == 'Quadrupole']
print('\nQuadrupoles:')
quadrupoles.cols['name s_start s_end'].show(maxrows = 12)

################################################################################
# Inspect the reference and strength variables
################################################################################
print('\nStrength variables:')
env.vars.get_table().rows['kqf|kqd'].show(maxrows = 12)
print(f'Circumference: {line.get_length():.3f} m')
print(f'Reference p0*c: {line.particle_ref.p0c[0] * 1e-9:.3f} GeV')

print(f'qf.1 k1: {line["qf.1"].k1:.9f} m^-2')
print(f'qf.2 k1: {line["qf.2"].k1:.9f} m^-2')
