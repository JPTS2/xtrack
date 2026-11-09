# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model and entrance optics
########################################
LATTICE_FILE    = 'test_data/example_lattices/transport/lattice.py'
LINE_NAME       = 'transport'
BETX_START      = 1.0                    # [m]
BETY_START      = 1.0                    # [m]
ALFX_START      = 0.0
ALFY_START      = 0.0

########################################
# Exit targets and matching settings
########################################
BETX_TARGET     = 50.0                   # [m]
BETY_TARGET     = 45.0                   # [m]
BETA_TOL        = 1.0e-7                 # [m]
K1_STEP         = 1.0e-5                 # [m^-2]

################################################################################
# Load the transport and propagate specified entrance optics
################################################################################
# Electron reference p0*c = 1 GeV; no RF or radiation. This is open-line optics:
# the entrance conditions stay fixed, and no periodic tune is being matched.
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
init        = xt.TwissInit(
    betx    = BETX_START,
    bety    = BETY_START,
    alfx    = ALFX_START,
    alfy    = ALFY_START,
)
tw_before   = line.twiss4d(init = init)

################################################################################
# Match the beta functions at the actual line exit
################################################################################
# xt.END selects the endpoint AFTER the last element, including the final drift.
# The two variables drive the transport's three quadrupoles through families.
# These targets constrain exit betas, not exit alphas or dispersion.
opt = line.match(
    method  = '4d',
    init    = init,
    vary    = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets = xt.TargetSet(
        betx    = BETX_TARGET,
        bety    = BETY_TARGET,
        at      = xt.END,
        tol     = BETA_TOL,
    ),
)
tw_after    = line.twiss4d(init = init)

################################################################################
# Compare the exit values and the full transport optics
################################################################################

########################################
# Inspect the actual exit and target values
########################################
print('Actual exit s [m]:', tw_after.s[-1])
print(f'{"Exit quantity":<16} {"Before [m]":>14} {"After [m]":>14}'
      f' {"Target [m]":>14}')
for quantity, target in [('betx', BETX_TARGET), ('bety', BETY_TARGET)]:
    print(f'{quantity:<16} {tw_before[quantity][-1]:14.8f}'
          f' {tw_after[quantity][-1]:14.8f} {target:14.8f}')
print('Matched family strengths [m^-2]:', env['kqf'], env['kqd'])

########################################
# Compare beta functions along the transport
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for axis, quantity, target in zip(
    axes, ['betx', 'bety'], [BETX_TARGET, BETY_TARGET]
):
    axis.plot(tw_before.s, tw_before[quantity], label = 'Before')
    axis.plot(tw_after.s, tw_after[quantity], label = 'Matched')
    axis.plot(tw_after.s[-1], target, 'kx', label = 'Exit target')
    axis.set_ylabel(quantity + ' [m]')
    axis.legend()
    axis.grid()
axes[1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
