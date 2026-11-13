# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import xtrack as xt


################################################################################
# User parameters
################################################################################

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

K1_STEP         = 1e-5                  # Gradient response step [m^-2]

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
transport               = xt.Environment(
    particle_ref = xt.Particles('electron', p0c = P0C_TRANSPORT),
)
transport['k_common']   = K1_COMMON
transport['k_a']        = K1_ROUTE_A
transport['k_b']        = K1_ROUTE_B
transport.new('q.common', xt.Quadrupole, length = 0.3, k1 = 'k_common')
transport.new('q.a', xt.Quadrupole, length = 0.3, k1 = 'k_a')
transport.new('q.b', xt.Quadrupole, length = 0.3, k1 = 'k_b')
transport.new('drift.2', xt.Drift, length = 2.0)
transport.new('drift.3', xt.Drift, length = 3.0)
transport.new('drift.4', xt.Drift, length = 4.0)
transport['route_a']    = transport.new_line(
    components = ['q.common', 'drift.2', 'q.a', 'drift.3'],
)
transport['route_b']    = transport.new_line(
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
