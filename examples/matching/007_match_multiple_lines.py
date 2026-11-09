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
# Two ring configurations with one shared family variable
########################################
LATTICE_FILE    = 'test_data/example_lattices/ring/lattice.py'
LINE_NAMES      = ['a', 'b']
QX_A_TARGET     = 1.70                  # Full tunes, including integer part
QY_A_TARGET     = 1.65
QX_B_TARGET     = 1.72

########################################
# Matching precision
########################################
K1_STEP         = 1e-5                  # Strength finite-difference step [m^-2]
TUNE_TOL        = 1e-8

################################################################################
# Prepare independent magnets with explicitly shared strength dependencies
################################################################################

########################################
# Give each source its own defocusing-family variable
########################################
repo_root   = Path(__file__).resolve().parents[2]
sources     = {}
for label in LINE_NAMES:
    source  = xt.load(repo_root / LATTICE_FILE)
    line    = source['ring']
    knob    = 'kqd_' + label
    source[knob] = source['kqd']
    for name in line.get_table().rows['qd[.].*'].name:
        source.set(name, k1 = knob)
    sources[label] = line

########################################
# Import both lines into one matching environment
########################################
# The magnets are independent copies; their colliding names are renamed on
# import. Equal variable names remain shared: both focusing families use kqf.
# The defocusing families instead depend on kqd_a and kqd_b, independently.

# Both lines have the native 40 m geometry and 1 GeV/c electron reference;
# they are two configurations, not counter-rotating beams; no RF/radiation.
env         = xt.Environment(lines = sources)
before      = {name: env[name].twiss4d() for name in LINE_NAMES}

################################################################################
# Solve three conditions jointly with three independent variables
################################################################################
# Matching line a fixes its two tunes. The third variable lets line b meet its
# horizontal tune while respecting the shared kqf. Its vertical tune is free:
# four independent tune requests would require another independent control.
opt = env.match(
    lines   = LINE_NAMES,
    method  = '4d',
    vary    = xt.VaryList(['kqf', 'kqd_a', 'kqd_b'], step = K1_STEP),
    targets = [
        xt.TargetSet(
            qx      = QX_A_TARGET,
            qy      = QY_A_TARGET,
            line    = 'a',
            tol     = TUNE_TOL,
        ),
        xt.Target(
            'qx',
            value   = QX_B_TARGET,
            line    = 'b',
            tol     = TUNE_TOL,
        ),
    ],
)

################################################################################
# Recompute each physical line and inspect the shared solution
################################################################################

########################################
# Compare both sets of tunes
########################################
after       = {name: env[name].twiss4d() for name in LINE_NAMES}
print(f'{"Line":<6} {"Qx before":>14} {"Qx after":>14}'
      f' {"Qy before":>14} {"Qy after":>14}')
for name in LINE_NAMES:
    print(f'{name:<6} {before[name].qx:14.9f} {after[name].qx:14.9f}'
          f' {before[name].qy:14.9f} {after[name].qy:14.9f}')
print('Targets: a Qx/Qy =', QX_A_TARGET, QY_A_TARGET, '; b Qx =', QX_B_TARGET)
print('Line b Qy is reported, not constrained.')

########################################
# Inspect the common and independent family variables
########################################
for knob in ['kqf', 'kqd_a', 'kqd_b']:
    print(knob, '[m^-2]:', env[knob])

# env.match evaluates both lines for one common variable state; solving them
# separately could overwrite the shared strength chosen for the other line.
