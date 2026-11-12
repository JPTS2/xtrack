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
# Ring and physical tune knobs
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
KNOB_NAME       = 'horizontal_tune'
VERTICAL_KNOB   = 'vertical_tune'
QX_INCREMENT    = 0.002                 # Matched endpoint Qx increment
QY_INCREMENT    = 0.001                 # Matched endpoint Qy increment
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
    knob_value_start    = reference.qx,
    knob_value_end      = reference.qx + QX_INCREMENT,
    run                 = False,
    method              = '4d',
    vary                = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets             = xt.TargetSet(
        qx      = reference.qx + QX_INCREMENT,
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
# Compose a second knob on the same quadrupole families
################################################################################

########################################
# Match a vertical-tune endpoint at the horizontal origin
########################################
vertical_opt = line.match_knob(
    knob_name           = VERTICAL_KNOB,
    knob_value_start    = reference.qy,
    knob_value_end      = reference.qy + QY_INCREMENT,
    run                 = False,
    method              = '4d',
    vary                = xt.VaryList(['kqf', 'kqd'], step = K1_STEP),
    targets             = xt.TargetSet(
        qx      = reference.qx,
        qy      = reference.qy + QY_INCREMENT,
        tol     = TUNE_TOL,
    ),
)
vertical_opt.solve()
vertical_opt.generate_knob()

########################################
# Inspect the composed strength expressions
########################################
# Both knobs have physical, nonzero origins. Each correction vanishes at its
# own origin, so the two generated corrections add to the saved strengths.
for name in k1_start:
    print(name, env.vars.get_expr(name))
    for knob_name in [KNOB_NAME, VERTICAL_KNOB]:
        correction = f'{name}_from_{knob_name}'
        print(correction, env.vars.get_expr(correction))
print('Horizontal/vertical origins:', env[KNOB_NAME], env[VERTICAL_KNOB])

################################################################################
# Recompute the physical optics at endpoints and intermediate settings
################################################################################
# Strengths interpolate linearly; tunes generally do not. Only the end setting
# was matched. Intermediate Qx error and Qy leakage quantify this finite-range
# approximation. No hardware limits or safe extrapolation range are established.
settings    = np.linspace(0, QX_INCREMENT, N_SAMPLES)
qx_values   = []
qy_values   = []
print(f'{"Qx increment":>14} {"Qx offset":>14}'
      f' {"Qx error":>14} {"Qy leakage":>14}')
for value in settings:
    env[KNOB_NAME] = reference.qx + value
    tw = line.twiss4d()
    qx_values.append(tw.qx)
    qy_values.append(tw.qy)
    print(f'{value:14.6f} {tw.qx - reference.qx:14.9f}'
          f' {tw.qx - reference.qx - value:14.3e}'
          f' {tw.qy - reference.qy:14.3e}')
qx_values   = np.array(qx_values)
qy_values   = np.array(qy_values)
qx_error    = qx_values - reference.qx - settings
qy_leakage  = qy_values - reference.qy

################################################################################
# Check separate endpoints, a mixed setting, and the common origin
################################################################################

########################################
# Set both physical tune values and recompute the optics
########################################
# The two endpoints were matched separately. Their simultaneous use need not
# give exact tune addition because optics is nonlinear in quadrupole strengths.
for dx, dy in [(QX_INCREMENT, 0), (0, QY_INCREMENT),
               (QX_INCREMENT, QY_INCREMENT), (0, 0)]:
    env[KNOB_NAME]       = reference.qx + dx
    env[VERTICAL_KNOB]   = reference.qy + dy
    checked = line.twiss4d()
    print('Requested Qx/Qy:', env[KNOB_NAME], env[VERTICAL_KNOB])
    print('Recomputed Qx/Qy:', checked.qx, checked.qy)
    print('Calibration residuals:', checked.qx - env[KNOB_NAME],
          checked.qy - env[VERTICAL_KNOB])

########################################
# Verify the reset recovers the original family strengths
########################################
for name, value in k1_start.items():
    print(name, 'reset residual [m^-2]:', env[name] - value)

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
axes[1].set_xlabel('Horizontal tune knob increment from its origin')
axes[1].legend()
for axis in axes:
    axis.grid()
fig.tight_layout()
plt.show()
