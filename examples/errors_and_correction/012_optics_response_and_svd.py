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
# Fixed quadrupole errors and independent trims
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
SEED            = 73
K1_ERROR_RMS    = 1e-5                  # Fixed additive errors [m^-2]
K1_STEP         = 1e-7                  # Response difference step [m^-2]

########################################
# Observable scaling and retained singular values
########################################
BETA_SCALE      = 1e-3                  # Relative beta-beating scale
TUNE_SCALE      = 1e-4                  # Tune-error scale
RCOND           = 1e-3
N_ITERATIONS    = 2

################################################################################
# Load nominal optics and apply independent fixed errors
################################################################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
quad_names  = [f'arc.s{s}.c{c}.qd' for s in range(1, 5) for c in [1, 2]]
trim_names  = [f'dq_s{s}_c{c}' for s in range(1, 5) for c in [1, 2]]
error_names = [f'eq_s{s}_c{c}' for s in range(1, 5) for c in [1, 2]]
reference   = line.twiss4d()
rng         = np.random.default_rng(SEED)
for name in error_names:
    env[name] = rng.normal(scale = K1_ERROR_RMS)
# Fixed errors are not correction variables. Both sets of controls already
# belong to the native DEMO quadrupole expressions; RF and radiation are off.

################################################################################
# Define one scaled optics measurement vector
################################################################################
def optics_vector(tw):
    return np.concatenate([
        (tw['betx', quad_names] / reference['betx', quad_names] - 1)
        / BETA_SCALE,
        (tw['bety', quad_names] / reference['bety', quad_names] - 1)
        / BETA_SCALE,
        [(tw.qx - reference.qx) / TUNE_SCALE,
         (tw.qy - reference.qy) / TUNE_SCALE],
    ])

before = line.twiss4d()

################################################################################
# Measure the response, inspect its rank and apply a truncated-SVD step
################################################################################

########################################
# Remeasure the response after each correction
########################################
for iteration in range(N_ITERATIONS):
    measured    = optics_vector(line.twiss4d())
    response    = np.zeros((len(measured), len(trim_names)))
    for column, name in enumerate(trim_names):
        value               = env[name]
        env[name]           = value + K1_STEP
        plus                = optics_vector(line.twiss4d())
        env[name]           = value - K1_STEP
        minus               = optics_vector(line.twiss4d())
        env[name]           = value
        response[:, column] = (plus - minus) / (2 * K1_STEP)

    left, singular_values, right = np.linalg.svd(
        response, full_matrices = False)
    retained = singular_values > RCOND * singular_values[0]
    amplitudes = -(left[:, retained].T @ measured) / singular_values[retained]
    correction = right[retained].T @ amplitudes
    for name, change in zip(trim_names, correction):
        env[name] += change
    print('Iteration:', iteration + 1)
    print('Singular values:', singular_values)
    print('Retained rank:', np.count_nonzero(retained), '/', len(trim_names))
    print('Scaled residual norm before/after:', np.linalg.norm(measured),
          np.linalg.norm(optics_vector(line.twiss4d())))
# Scaling defines the least-squares objective. A cutoff removes poorly observed
# directions; it does not identify a unique physical error distribution.

################################################################################
# Inspect full-ring residuals outside the measured vector
################################################################################
after = line.twiss4d()
print('Final Qx/Qy residuals:', after.qx - reference.qx,
      after.qy - reference.qy)
fig, axes = plt.subplots(3, 1, figsize = (9, 9),
                         layout = 'constrained')
for axis, quantity in zip(axes, ['betx', 'bety']):
    for label, tw in [('Before', before), ('After', after)]:
        axis.plot(tw.s, 100 * (tw[quantity] / reference[quantity] - 1),
                  label = label)
    axis.set_ylabel(quantity + ' beating [%]')
    axis.grid()
    axis.legend()
axes[1].set_xlabel('s [m]')
axes[2].semilogy(np.arange(1, len(singular_values) + 1),
                 singular_values, 'o-', label = 'Final response spectrum')
axes[2].axhline(RCOND * singular_values[0], color = 'grey', linestyle = ':',
                label = 'Relative cutoff')
axes[2].set(xlabel = 'Singular-value index',
             ylabel = 'Scaled response / k1 trim [m²]')
axes[2].grid()
axes[2].legend()
plt.show()
