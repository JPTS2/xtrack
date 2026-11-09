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
# Model and integration choices by class
########################################
# Lattice file relative to the repository root.
LATTICE_FILE        = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME           = 'demo'
BEND_MODEL          = 'rot-kick-rot'
BEND_INTEGRATOR     = 'teapot'
BEND_KICKS          = 4
QUAD_MODEL          = 'mat-kick-mat'
QUAD_INTEGRATOR     = 'yoshida-6'
QUAD_KICKS          = 7

########################################
# Focusing-family and individual overrides
########################################
FAMILY_MODEL        = 'drift-kick-drift-exact'
FAMILY_INTEGRATOR   = 'yoshida-6'
FAMILY_KICKS        = 21
SINGLE_ELEMENT      = 'arc.s1.c1.qf.in'
SINGLE_MODEL        = 'mat-kick-mat'
SINGLE_INTEGRATOR   = 'teapot'
SINGLE_KICKS        = 10

################################################################################
# Inspect the available choices and baseline optics
################################################################################

########################################
# Query public model and integrator names
########################################
# The model selects the body-map formulation. The integrator controls how its
# elementary maps and multipole kicks are composed. Their roles are distinct;
# num_multipole_kicks sets the requested kick count for that integration.
# Kick settings apply where the model needs thin kicks. Yoshida-6 rounds the
# count up to complete seven-kick slices (the choices below use 7 and 21).
for element_class in [xt.Bend, xt.Quadrupole]:
    print(f'{element_class.__name__} models:')
    print(element_class.get_available_models())
    print(f'{element_class.__name__} integrators:')
    print(element_class.get_available_integrators())

########################################
# Load the ring and calculate the baseline
########################################
repo_root       = Path(__file__).resolve().parents[3]
env             = xt.load(repo_root / LATTICE_FILE)
line            = env[LINE_NAME]

# The model supplies an electron reference particle with p0*c = 1 GeV.
table_before    = line.get_table(attr = True)
tw0             = line.twiss(method = '4d')

################################################################################
# Configure maps by class, family and individual element
################################################################################

########################################
# Select element classes from the line table
########################################
bends       = table_before.rows[
    (table_before.element_type == 'Bend')
    | (table_before.element_type == 'RBend')
]
quadrupoles = table_before.rows[table_before.element_type == 'Quadrupole']
line.set(
    bends,
    model               = BEND_MODEL,
    integrator          = BEND_INTEGRATOR,
    num_multipole_kicks = BEND_KICKS,
)
line.set(
    quadrupoles,
    model               = QUAD_MODEL,
    integrator          = QUAD_INTEGRATOR,
    num_multipole_kicks = QUAD_KICKS,
)

########################################
# Override the focusing quadrupole family
########################################
# A name regex selects only arc QF halves, excluding insertion quadrupoles.
focusing = quadrupoles.rows[r'arc[.]s[1-6][.]c[1-4][.]qf[.].*']
line.set(
    focusing,
    model               = FAMILY_MODEL,
    integrator          = FAMILY_INTEGRATOR,
    num_multipole_kicks = FAMILY_KICKS,
)

########################################
# Override one focusing quadrupole
########################################
line[SINGLE_ELEMENT].model               = SINGLE_MODEL
line[SINGLE_ELEMENT].integrator          = SINGLE_INTEGRATOR
line[SINGLE_ELEMENT].num_multipole_kicks = SINGLE_KICKS
print('Configured body-map fields (name, model, integrator, kick count):')
for name in ['arc.s1.c1.b1', 'arc.s1.c1.qd', 'arc.s1.c2.qf.in', SINGLE_ELEMENT]:
    element = line[name]
    print(name, element.model, element.integrator, element.num_multipole_kicks)

################################################################################
# Compare the configured optics with the baseline
################################################################################

########################################
# Check geometry and integrated quadrupole strengths
########################################
table_after = line.get_table(attr = True)
for column, unit in [('s_start', 'm'), ('s_end', 'm'), ('angle', 'rad'),
                     ('k1l', 'm^-1')]:
    difference = np.max(np.abs(table_after[column] - table_before[column]))
    print(f'Maximum change in {column}: {difference:.3e} {unit}')

########################################
# Recalculate four-dimensional periodic optics
########################################
# Strengths and geometry stay fixed, but different map approximations can give
# different optics. This comparison is not an accuracy or speed ranking.
tw = line.twiss(method = '4d')
print(f'Baseline tunes:   Qx = {tw0.qx:.9f}, Qy = {tw0.qy:.9f}')
print(f'Configured tunes: Qx = {tw.qx:.9f}, Qy = {tw.qy:.9f}')
print(f'Tune changes: dQx = {tw.qx - tw0.qx:+.3e}, '
      f'dQy = {tw.qy - tw0.qy:+.3e}')
