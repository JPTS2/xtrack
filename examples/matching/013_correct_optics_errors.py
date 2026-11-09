# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ring and fixed normal-quadrupole errors
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SEED            = 73
K1_ERROR_RMS    = 1e-5                  # Additive k1-error RMS [m^-2]

########################################
# Independent correction trims and optics targets
########################################
K1_STEP         = 1e-7                  # Strength finite-difference step [m^-2]
BETA_TOL        = 1e-6                  # Sampled beta-function tolerance [m]
TUNE_TOL        = 1e-8

################################################################################
# Set up fixed errors and separate trim controls at eight arc quadrupoles
################################################################################

########################################
# Preserve the nominal family strength dependencies
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
quad_names  = []
trim_names  = []
error_names = []
for sector in range(1, 5):
    for cell in [1, 2]:
        quad_names.append(f'arc.s{sector}.c{cell}.qd')
        trim_names.append(f'dq_s{sector}_c{cell}')
        error_names.append(f'eq_s{sector}_c{cell}')

# The 1500 m ring has a 1 GeV/c electron reference, with RF voltage zero and
# radiation off.
reference = line.twiss4d()

########################################
# Apply reproducible errors without changing the trims
########################################
rng = np.random.default_rng(SEED)
for name in error_names:
    env[name] = rng.normal(scale = K1_ERROR_RMS)
perturbed = line.twiss4d()

################################################################################
# Infer correction trims from beta functions and tunes
################################################################################

########################################
# Define reference beta and tune targets
########################################
# This ideal noiseless example has an independent trim at each selected
# error location.
# Errors stay fixed and are NOT varied by the optimizer. The matching targets
# use reference optics, not the known random error values. Real sparse-actuator
# or noisy reconstructions need a separate response/rank/regularization study.
targets = []
for name in quad_names:
    for quantity in ['betx', 'bety']:
        targets.append(xt.Target(
            quantity,
            at      = name,
            value   = reference,
            tol     = BETA_TOL,
        ))
targets.append(xt.TargetSet(
    qx  = reference.qx,
    qy  = reference.qy,
    tol = TUNE_TOL,
))

########################################
# Solve independent trim strengths
########################################
opt = line.match(
    method  = '4d',
    vary    = xt.VaryList(trim_names, step = K1_STEP),
    targets = targets,
)
corrected = line.twiss4d()

################################################################################
# Inspect sampled residuals and the full-ring beta beating
################################################################################

########################################
# Report beta-beating RMS at quadrupole entrances and tune errors
########################################
for label, tw in [('Perturbed', perturbed), ('Corrected', corrected)]:
    print(label, 'Qx/Qy residuals:', tw.qx - reference.qx, tw.qy - reference.qy)
    for quantity in ['betx', 'bety']:
        sampled = tw[quantity, quad_names] / reference[quantity, quad_names] - 1
        print(label, quantity, 'sampled relative RMS:',
              np.sqrt(np.mean(sampled**2)))
print('Fixed-error and fitted-trim RMS [m^-2]:',
      np.sqrt(np.mean([env[name]**2 for name in error_names])),
      np.sqrt(np.mean([env[name]**2 for name in trim_names])))

# Agreement of these observables does not generally identify a unique error set.

########################################
# Compare the beta beating throughout the ring
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for axis, quantity in zip(axes, ['betx', 'bety']):
    for label, tw in [('Perturbed', perturbed), ('Corrected', corrected)]:
        axis.plot(
            tw.s, 100 * (tw[quantity] / reference[quantity] - 1),
            label   = label,
        )
    axis.set_ylabel(quantity + ' beating [%]')
    axis.grid()
    axis.legend()
axes[-1].set_xlabel('s [m]')
fig.tight_layout()
plt.show()
