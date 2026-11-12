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
# Two ring configurations with one shared varied family
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAMES      = ['a', 'b']
QX_A_TARGET     = 8.760                  # Full tunes, including integer part
QY_A_TARGET     = 10.020
QX_B_TARGET     = 8.765

########################################
# Matching precision
########################################
K1_STEP         = 1e-5                  # Strength finite-difference step [m^-2]
TUNE_TOL        = 1e-8

########################################
# Shared-magnet transport reference
########################################
P0C_TRANSPORT   = 1e9                   # Electron reference p0*c [eV]
BETX_ENTRANCE   = 3.0                   # Both routes [m]; alpha_x = 0
BETY_ENTRANCE   = 4.0                   # Both routes [m]; alpha_y = 0
K1_COMMON       = 0.20                  # Shared quadrupole [m^-2]
K1_ROUTE_A      = -0.30                 # Independent quadrupole [m^-2]
K1_ROUTE_B      = -0.25                 # Independent quadrupole [m^-2]
K1_ERRORS       = [0.02, -0.015, 0.01]  # Common, route a, route b [m^-2]
BETA_TOL        = 1e-7                  # Exit beta targets [m]

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
    line    = source['demo']
    knob    = 'kqd_' + label
    source[knob] = source['kqd']
    for name in line.get_table().rows['arc[.]s[1-6][.]c[1-4][.]qd'].name:
        source.set(name, k1 = knob)
    sources[label] = line

########################################
# Import both lines into one matching environment
########################################
# The magnets are independent copies; their colliding names are renamed on
# import. Equal variable names remain shared: both focusing families use kqf.
# The defocusing families instead depend on kqd_a and kqd_b, independently.

# Both lines have the native 1500 m geometry and 1 GeV/c electron reference;
# they are two configurations, with RF voltage zero and radiation off.
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

########################################
# Distinguish common settings from a shared element
########################################
qf_a = env['a'].element_names[0]
qf_b = env['b'].element_names[0]
print('Independent focusing element names:', qf_a, qf_b)
print('The two configurations share this element object:',
      env['a'].get(qf_a) is env['b'].get(qf_b))
print('Their strengths follow the same kqf [m^-2]:',
      env['a'][qf_a].k1, env['b'][qf_b].k1)

# env.match evaluates both lines for one common variable state; solving them
# separately could overwrite the shared strength chosen for the other line.

################################################################################
# Match two transport routes through one physical quadrupole
################################################################################

########################################
# Build shared and independent elements
########################################
# Two forward-going electron routes share their entrance quadrupole, then
# have different downstream optics. Both use the same local axes and rigidity:
# positive k1 focuses x in both routes. No beam reversal or field sign flip
# is involved. The routing switch and global branch geometry are not modelled.
transport = xt.Environment(
    particle_ref = xt.Particles('electron', p0c = P0C_TRANSPORT),
)
transport['k_common']    = K1_COMMON
transport['k_a']         = K1_ROUTE_A
transport['k_b']         = K1_ROUTE_B
transport.new('q.common', xt.Quadrupole, length = 0.3, k1 = 'k_common')
transport.new('q.a', xt.Quadrupole, length = 0.3, k1 = 'k_a')
transport.new('q.b', xt.Quadrupole, length = 0.3, k1 = 'k_b')
transport.new('drift.2', xt.Drift, length = 2.0)
transport.new('drift.3', xt.Drift, length = 3.0)
transport.new('drift.4', xt.Drift, length = 4.0)
transport['route_a'] = transport.new_line(
    components = ['q.common', 'drift.2', 'q.a', 'drift.3'],
)
transport['route_b'] = transport.new_line(
    components = ['q.common', 'drift.3', 'q.b', 'drift.4'],
)
print('Both routes contain the same quadrupole object:',
      transport['route_a'].get('q.common')
      is transport['route_b'].get('q.common'))

########################################
# Record nominal optics and introduce strength errors
########################################
route_names = ['route_a', 'route_b']
entrances   = [xt.TwissInit(betx = BETX_ENTRANCE, bety = BETY_ENTRANCE)
              for name in route_names]
reference   = transport.twiss(
    lines   = route_names,
    method  = '4d',
    init    = entrances,
)
knobs       = ['k_common', 'k_a', 'k_b']
for knob, error in zip(knobs, K1_ERRORS):
    transport[knob] += error

########################################
# Combine individual and joint targets
########################################
# Restore both exit betas in route a and the SUM of horizontal exit betas.
# The callable sees both Twiss tables at the same magnet settings. Together
# these three conditions also restore route b's exit betx; its bety is free.
shared_opt = transport.match(
    lines   = route_names,
    method  = '4d',
    init    = entrances,
    vary    = xt.VaryList(knobs, step = K1_STEP, limits = [-1.0, 1.0]),
    targets = [
        xt.TargetSet(
            betx    = reference.route_a.betx[-1],
            bety    = reference.route_a.bety[-1],
            at      = xt.END,
            line    = 'route_a',
            tol     = BETA_TOL,
        ),
        xt.Target(
            lambda tw: tw.route_a.betx[-1] + tw.route_b.betx[-1],
            value   = reference.route_a.betx[-1] + reference.route_b.betx[-1],
            tol     = BETA_TOL,
        ),
    ],
)

########################################
# Recompute both routes with the shared setting
########################################
shared_after = transport.twiss(
    lines   = route_names,
    method  = '4d',
    init    = entrances,
)
for name in route_names:
    print(name, 'reference exit betx/bety [m]:',
          reference[name].betx[-1], reference[name].bety[-1])
    print(name, 'matched exit betx/bety [m]:',
          shared_after[name].betx[-1], shared_after[name].bety[-1])
print('Joint betx-sum residual [m]:',
      shared_after.route_a.betx[-1] + shared_after.route_b.betx[-1]
      - reference.route_a.betx[-1] - reference.route_b.betx[-1])
print('Common quadrupole k1 seen by both routes [m^-2]:',
      transport['route_a']['q.common'].k1,
      transport['route_b']['q.common'].k1)
print('Route b bety and both exit alpha functions are not targets.')
