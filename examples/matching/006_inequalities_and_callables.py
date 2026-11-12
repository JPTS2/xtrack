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
KQF_FACTOR      = 1.0002                # Perturb only the focusing family
QX_HALF_WIDTH   = 0.01                  # Allowed interval around reference Qx

########################################
# Numerical matching settings
########################################
K1_STEP         = 1.0e-5                # Finite-difference step [m^-2]
KQF_LIMITS      = [0.06, 0.08]          # Numerical bounds [m^-2]
KQD_LIMITS      = [-0.08, -0.06]        # Numerical bounds [m^-2]
TUNE_TOL        = 1.0e-8
QF_WEIGHTS      = [1.0, 10.0]           # Relative control scale
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
# Default step inequalities give zero penalty inside the allowed interval.
# They do not request either boundary as an equality or impose hard feasibility.
# Strength bounds below are numerical search bounds, not hardware limits.
opt = line.match(
    solve   = False,
    method  = '4d',
    vary    = [
        xt.Vary('kqf', step = K1_STEP, limits = KQF_LIMITS),
        xt.Vary('kqd', step = K1_STEP, limits = KQD_LIMITS),
    ],
    targets = [
        xt.Target(
            'qx', xt.GreaterThan(qx_lower),
            tol = TUNE_TOL,
        ),
        xt.Target(
            'qx', xt.LessThan(qx_upper),
            tol = TUNE_TOL,
        ),
        xt.Target(
            lambda tw: tw['betx', OBSERVATION] / tw['bety', OBSERVATION],
            value   = ratio_ref,
            tol     = RATIO_TOL,
            tag     = 'beta_ratio',
        ),
    ],
)

################################################################################
# Freeze a target at its current value, then restore its requested value
################################################################################

########################################
# Hold the perturbed ratio temporarily
########################################
# freeze() changes the target value to the current observable. It does not
# disable the constraint or hold either quadrupole family fixed.
opt.targets[2].freeze()
print('Frozen beta-ratio target:', opt.targets[2].value)
opt.target_status()

########################################
# Recover the original request and solve
########################################
opt.targets[2].unfreeze()
print('Restored beta-ratio target:', opt.targets[2].value)
opt.solve()

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

################################################################################
# Allocate an underdetermined correction with variable weights
################################################################################
# One ratio equality leaves a free direction while the tune bounds are inactive.
# A larger Vary weight favors that family in the scaled minimum-norm step.
# Target weights rescale residuals; they do not replace tolerances.
# This repetition also selects smooth inequality penalties: their transition
# into the forbidden region is gradual instead of the default linear penalty.

initial_strengths = opt.get_knob_values(iteration = 0)
for qf_weight in QF_WEIGHTS:
    for name, value in initial_strengths.items():
        env[name] = value
    weighted = line.match(
        method  = '4d',
        vary    = [
            xt.Vary(
                'kqf',
                step    = K1_STEP,
                weight  = qf_weight,
                limits  = KQF_LIMITS,
            ),
            xt.Vary(
                'kqd',
                step    = K1_STEP,
                weight  = 1.0,
                limits  = KQD_LIMITS,
            ),
        ],
        targets = [
            xt.Target(
                'qx', xt.GreaterThan(qx_lower, mode = 'smooth'),
                tol = TUNE_TOL,
            ),
            xt.Target(
                'qx', xt.LessThan(qx_upper, mode = 'smooth'),
                tol = TUNE_TOL,
            ),
            xt.Target(
                lambda tw: tw['betx', OBSERVATION] / tw['bety', OBSERVATION],
                value   = ratio_ref,
                tol     = RATIO_TOL,
                weight  = 1.0,
            ),
        ],
    )
    checked = line.twiss4d()
    print('Focusing-family weight:', qf_weight)
    for name, value in initial_strengths.items():
        print(name, 'change [m^-2]:', env[name] - value)
    print('Ratio residual:', checked['betx', OBSERVATION]
          / checked['bety', OBSERVATION] - ratio_ref)
    print('Distances from Qx bounds:', checked.qx - qx_lower,
          qx_upper - checked.qx)
