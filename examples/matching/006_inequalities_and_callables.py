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
# Reference optics and deliberate perturbation
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
OBSERVATION     = 'bpm.s1.f'            # Passive read plane at s=1 m
KQF_FACTOR      = 1.0002                  # Perturb only the focusing family
QX_HALF_WIDTH   = 0.01                  # Allowed interval around reference Qx

########################################
# Numerical matching settings
########################################
K1_STEP         = 1.0e-5                # Finite-difference step [m^-2]
KQF_LIMITS      = [0.06, 0.08]             # Numerical bounds [m^-2]
KQD_LIMITS      = [-0.08, -0.06]           # Numerical bounds [m^-2]
TUNE_TOL        = 1.0e-8
RATIO_TOL       = 1.0e-8                # Dimensionless

################################################################################
# Capture the reference and define a tune interval
################################################################################

########################################
# Calculate the stated reference ratio
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# A 1500 m electron ring at p0*c=1 GeV, with RF voltage zero and radiation off.
# Optics here is periodic 4D at delta=0 and fixed reference momentum.
reference   = line.twiss4d()
ratio_ref   = (reference['betx', OBSERVATION]
               / reference['bety', OBSERVATION])
qx_lower    = reference.qx - QX_HALF_WIDTH
qx_upper    = reference.qx + QX_HALF_WIDTH

########################################
# Perturb a family strength before matching
########################################
env['kqf'] *= KQF_FACTOR
before = line.twiss4d()
print('Before: Qx and local betx/bety:', before.qx,
      before['betx', OBSERVATION] / before['bety', OBSERVATION])
print('Allowed Qx interval:', qx_lower, qx_upper)
print('Target betx/bety at', OBSERVATION, ':', ratio_ref)

################################################################################
# Combine inequality targets with a callable optics observable
################################################################################
# The callable receives the Twiss result and returns a dimensionless ratio.
# Default step inequalities contribute zero penalty inside their allowed region;
# they do not request either boundary as an equality or impose hard feasibility.
# Strength bounds below are numerical search bounds, not hardware limits.
opt = line.match(
    method  = '4d',
    vary    = [
        xt.Vary('kqf', step = K1_STEP, limits = KQF_LIMITS),
        xt.Vary('kqd', step = K1_STEP, limits = KQD_LIMITS),
    ],
    targets = [
        xt.Target('qx', xt.GreaterThan(qx_lower), tol = TUNE_TOL),
        xt.Target('qx', xt.LessThan(qx_upper), tol = TUNE_TOL),
        xt.Target(
            lambda tw: tw['betx', OBSERVATION] / tw['bety', OBSERVATION],
            value   = ratio_ref,
            tol     = RATIO_TOL,
            tag     = 'beta_ratio',
        ),
    ],
)

################################################################################
# Inspect physical values as well as target status
################################################################################

########################################
# Check the achieved tune interval and beta ratio
########################################
opt.target_status()
after           = line.twiss4d()
ratio_after     = after['betx', OBSERVATION] / after['bety', OBSERVATION]
print(f'After: Qx = {after.qx:.10f}, betx/bety = {ratio_after:.10f}')
print(f'Ratio residual: {ratio_after - ratio_ref:+.3e}')
print('Distances from lower/upper Qx limits:',
      after.qx - qx_lower, qx_upper - after.qx)

########################################
# Inspect the unconstrained optics and strengths
########################################
print('Reference/final Qy:', reference.qy, after.qy)
for name in ['kqf', 'kqd']:
    print(name, '[m^-2]:', env[name])

# The interval and ratio do not specify a unique lattice. Qy and other optics
# are not targets, and restoring this ratio does not restore the full reference.
