# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path
from textwrap import fill

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
print('Available body models and integrators'
      '\n-------------------------------------')
print('The model chooses the body map; the integrator composes its steps.')
print('num_multipole_kicks is a requested count, '
      'used where the model needs kicks.')
print('Yoshida-6 uses complete seven-kick slices; '
      'this example requests 7 or 21.')
for element_class in [xt.Bend, xt.Quadrupole]:
    print(f'\n{element_class.__name__} models:')
    print(fill(', '.join(element_class.get_available_models()), width = 78))
    print(f'{element_class.__name__} integrators:')
    print(fill(', '.join(element_class.get_available_integrators()),
               width = 78))

########################################
# Load the ring and calculate the baseline
########################################
repo_root       = Path(__file__).resolve().parents[3]
env             = xt.load(repo_root / LATTICE_FILE)
line            = env[LINE_NAME]

# The model supplies a 1 GeV/c electron reference; RF and radiation are off.
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
line[SINGLE_ELEMENT].model                  = SINGLE_MODEL
line[SINGLE_ELEMENT].integrator             = SINGLE_INTEGRATOR
line[SINGLE_ELEMENT].num_multipole_kicks    = SINGLE_KICKS
print('\nApply class settings, then family and individual overrides'
      '\n----------------------------------------------------------')
print('These are assignments in sequence: later settings replace earlier ones.')
print(f'Class selections: {len(bends)} bends, {len(quadrupoles)} quadrupoles.')
print(f'Family selection: {len(focusing)} arc QF halves; '
      'insertion quads excluded.')
print('Final settings for representative elements:')
print(f'{"Scope":<12} {"Element":<22} {"Model":<25}'
      f' {"Integrator":<12} {"Requested kicks":>15}')
for scope, name in [
    ('Bend class', 'arc.s1.c1.b1'),
    ('Quad class', 'arc.s1.c1.qd'),
    ('Quad class', 'straight.s1.r.q0'),
    ('QF family', 'arc.s1.c2.qf.in'),
    ('Individual', SINGLE_ELEMENT),
]:
    element = line[name]
    print(f'{scope:<12} {name:<22} {element.model:<25}'
          f' {element.integrator:<12} {element.num_multipole_kicks:15d}')

################################################################################
# Compare the configured optics with the baseline
################################################################################

########################################
# Check geometry and integrated quadrupole strengths
########################################
print('\nGeometry and integrated strengths stay fixed'
      '\n--------------------------------------------')
table_after = line.get_table(attr = True)
for column, unit in [('s_start', 'm'), ('s_end', 'm'), ('angle', 'rad'),
                     ('k0l', '1'), ('k1l', 'm^-1')]:
    difference = np.max(np.abs(table_after[column] - table_before[column]))
    print(f'Maximum change in {column} [{unit}]: {difference:.3e}')

########################################
# Recalculate four-dimensional periodic optics
########################################
# Strengths and geometry stay fixed, but different map approximations can give
# different optics. This comparison is not an accuracy or speed ranking.
print('\nCompare periodic 4D optics\n--------------------------')
print('Same strengths and geometry, different body-map settings.')
print('Tune differences alone do not establish an accuracy or speed ranking.')
tw = line.twiss(method = '4d')
print(f'Baseline tunes:   Qx = {tw0.qx:.9f}, Qy = {tw0.qy:.9f}')
print(f'Configured tunes: Qx = {tw.qx:.9f}, Qy = {tw.qy:.9f}')
print(f'Tune changes: dQx = {tw.qx - tw0.qx:+.3e}, '
      f'dQy = {tw.qy - tw0.qy:+.3e}')
