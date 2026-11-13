# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Further examples: examples/line_construction/

################################################################################
# Required packages
################################################################################
from pathlib import Path
from textwrap import fill

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

print('First 12 elements [m]\n---------------------')
# s_start and s_end are layout coordinates, in metres.
table.rows[:12].cols['name s_start s_end element_type'].show()

########################################
# Quadrupoles only
########################################
quadrupoles = table.rows[table.element_type == 'Quadrupole']
print('\nFirst 12 quadrupoles [m]\n------------------------')
quadrupoles.rows[:12].cols['name s_start s_end'].show()

########################################
# Attribute arrays in tracking order
########################################
# Arrays contain one value per element, without the table's final endpoint.
attributes = [name for name in line.attr.keys() if not name.startswith('_')]
print('\nAvailable public attributes\n---------------------------')
print(fill(', '.join(attributes), width = 78))
lengths = line.attr['length']
k1l     = line.attr['k1l']
print('\nAttribute arrays: first 12 elements'
      '\n----------------------------------')
print('Element lengths [m]:')
print(lengths[:12])
print('\nIntegrated quadrupole strengths [m^-1]:')
print(k1l[:12])

################################################################################
# Inspect the reference and strength variables
################################################################################
print('\nStrength variables\n------------------')
env.vars.get_table().rows['kqf|kqd'].show(maxrows = 12)
print('\nRing and reference\n------------------')
print(f'Circumference [m]: {line.get_length():.3f}')
print(f'Reference momentum [GeV/c]: {line.particle_ref.p0c[0] * 1e-9:.3f}')

print('\nElement strengths\n-----------------')
print(f'arc.s1.c1.qf.in k1 [m^-2]: {line["arc.s1.c1.qf.in"].k1:.9f}')
print(f'arc.s1.c2.qf.in k1 [m^-2]: {line["arc.s1.c2.qf.in"].k1:.9f}')
