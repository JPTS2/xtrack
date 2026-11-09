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
# Ring and requested horizontal tune-offset knob
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
KNOB_NAME       = 'horizontal_tune_offset'
KNOB_END        = 0.002                  # Matched endpoint Qx offset
N_SAMPLES       = 5

########################################
# Matching precision
########################################
K1_STEP         = 1e-5                  # Strength finite-difference step [m^-2]
TUNE_TOL        = 1e-8

################################################################################
# Match the endpoint correction and generate its expression dependencies
################################################################################

########################################
# Save the original periodic optics
########################################
repo_root   = Path(__file__).resolve().parents[2]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
# The 1500 m ring has a 1 GeV/c electron reference, with RF voltage zero and
# radiation off.
# All calculations are periodic 4D optics at fixed reference momentum, delta=0.
reference   = line.twiss4d()
k1_start    = {name: env[name] for name in ['kqf', 'kqd']}

########################################
# Solve the endpoint and install a reusable knob
########################################
# Two family corrections raise Qx while preserving Qy at this endpoint.
# Construction adds auxiliary corrections to the strength expressions.
# run=False lets us solve them before binding their values to the new knob.
opt = line.match_knob(
    knob_name           = KNOB_NAME,
    knob_value_start    = 0.0,
    knob_value_end      = KNOB_END,
    run                 = False,
    method              = '4d',
    vary                = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets             = xt.TargetSet(
        qx      = reference.qx + KNOB_END,
        qy      = reference.qy,
        tol     = TUNE_TOL,
    ),
)
opt.solve()
opt.generate_knob()

# generate_knob() makes the auxiliary corrections linear in the new knob and
# resets it to its starting value. It does not rerun a match per use.
print('Generated knob starting value:', env[KNOB_NAME])
print('Original/generated-start family strengths [m^-2]:')
for name, strength in k1_start.items():
    print(name, strength, env[name])

################################################################################
# Recompute the physical optics at endpoints and intermediate settings
################################################################################
# Strengths interpolate linearly; tunes generally do not. Only the end setting
# was matched. Intermediate Qx error and Qy leakage quantify this finite-range
# approximation. No hardware limits or safe extrapolation range are established.
settings    = np.linspace(0, KNOB_END, N_SAMPLES)
qx_values   = []
qy_values   = []
print(f'{"Knob":>10} {"Qx offset":>14} {"Qx error":>14} {"Qy leakage":>14}')
for value in settings:
    env[KNOB_NAME] = value
    tw = line.twiss4d()
    qx_values.append(tw.qx)
    qy_values.append(tw.qy)
    print(f'{value:10.6f} {tw.qx - reference.qx:14.9f}'
          f' {tw.qx - reference.qx - value:14.3e}'
          f' {tw.qy - reference.qy:14.3e}')
qx_values   = np.array(qx_values)
qy_values   = np.array(qy_values)
qx_error    = qx_values - reference.qx - settings
qy_leakage  = qy_values - reference.qy

################################################################################
# Plot the tune response and its departure from the ideal calibration
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
axes[0].plot(settings, qx_values - reference.qx, 'o-', label = 'Recomputed Qx')
axes[0].plot(settings, settings, 'k--', label = 'Ideal offset')
axes[0].set_ylabel('Qx offset')
axes[0].legend()
# The band is the endpoint solver tolerance, not a numerical error floor or
# an imposed constraint on intermediate settings. Keep the residuals visible.
axes[1].axhspan(-TUNE_TOL, TUNE_TOL, color = '0.8', alpha = 0.4,
                label   = 'Endpoint target tolerance (context)')
axes[1].plot(settings, qx_error, 'o-', label = 'Qx calibration error')
axes[1].plot(settings, qy_leakage, 's-', label = 'Qy leakage')
axes[1].set_ylabel('Tune residual')
axes[1].set_xlabel('Generated knob value (requested Qx offset)')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
