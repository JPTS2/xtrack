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
# Model and periodic tune targets
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
QX_TARGET       = 8.760                   # Full tune, including integer part
QY_TARGET       = 10.020

########################################
# Matching precision and variable step
########################################
TUNE_TOL        = 1e-8                   # Absolute tune tolerance
K1_STEP         = 1e-5                   # Finite-difference k1 step [m^-2]

################################################################################
# Inspect the initial periodic optics
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
# This 1500 m ring has a 1 GeV/c electron reference, with RF voltage zero
# and radiation off.
# Periodic 4D optics holds delta=0 at the fixed reference momentum.
tw_before   = line.twiss4d()
k1_before   = {name: env[name] for name in ['kqf', 'kqd']}

################################################################################
# Match two tunes with two quadrupole-family strengths
################################################################################
# kqf and kqd drive the focusing and defocusing ARC quadrupoles.
# Final-focus and utility-straight strengths remain fixed.
# match() solves immediately by default and leaves the changed strengths live.
opt = line.match(
    method  = '4d',
    vary    = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets = xt.TargetSet(qx = QX_TARGET, qy = QY_TARGET, tol = TUNE_TOL),
)
tw_after    = line.twiss4d()

################################################################################
# Compare the independently recomputed result with the requested tunes
################################################################################

########################################
# Inspect tune targets and residuals
########################################
print(f'{"Tune":<8} {"Before":>14} {"After":>14}'
      f' {"Target":>14} {"Residual":>12}')
for name, target in [('qx', QX_TARGET), ('qy', QY_TARGET)]:
    print(f'{name:<8} {tw_before[name]:14.9f} {tw_after[name]:14.9f}'
          f' {target:14.9f} {tw_after[name] - target:12.3e}')

########################################
# Inspect the family-strength changes
########################################
print('Family strengths before/after [m^-2]:')
for name, before in k1_before.items():
    print(f'{name}: {before:+.9f} -> {env[name]:+.9f}')

# Matching these two scalar observables does not restore the original beta
# functions, chromaticities, or any unrequested property of the ring.
