# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Native magnetic ring
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
LINE_NAME       = 'demo'
OBSERVATION     = 'arc.s1.c1.qf.in'

########################################
# Momentum scan and polynomial fits
########################################
DELTA_MAX       = 2e-5
NUM_DELTA       = 13                     # More samples than fit coefficients
FIT_ORDER       = 3                      # At least 2; leave room for order + 1
LOCAL_STEP      = 1e-5                   # Local Twiss derivative step
# The bare vertical tune is near an integer; use a small stable fit window.

################################################################################
# Fit the momentum dependence using the public helper
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env[LINE_NAME]
# The helper performs 4D Twiss scans at fixed reference momentum and fields.
# chrom=False avoids another chromatic derivative calculation at every sample.
fitted = line.get_non_linear_chromaticity(
    delta0_range    = (-DELTA_MAX, DELTA_MAX),
    num_delta       = NUM_DELTA,
    fit_order       = FIT_ORDER,
    chrom           = False)

################################################################################
# Interpret derivatives and reconstruct the fitted tunes
################################################################################

########################################
# Read the fit derivatives at delta = 0
########################################
# dnqx[n] is d^n Qx/d(delta)^n at zero, including the factorial. Order zero
# is the fitted intercept, not an independently calculated on-momentum tune.
orders = np.arange(FIT_ORDER + 1)
xt.Table(
    data    = {
        'order': orders,
        'dnqx':  fitted.dnqx,
        'dnqy':  fitted.dnqy},
    index   = 'order').show(maxwidth = 'full')

########################################
# Convert derivatives to ascending polynomial coefficients
########################################
# Q(delta) = sum_n dnQ[n] * delta**n / n!: the quadratic coefficient is Q''/2.
factorials  = np.array([factorial(int(n)) for n in orders])
qx_coeff    = fitted.dnqx / factorials
qy_coeff    = fitted.dnqy / factorials
qx_fit      = np.polynomial.polynomial.polyval(fitted.delta0, qx_coeff)
qy_fit      = np.polynomial.polynomial.polyval(fitted.delta0, qy_coeff)
print('Largest sampled tune-fit residuals (x, y):',
      np.max(np.abs(fitted.qx - qx_fit)), np.max(np.abs(fitted.qy - qy_fit)))

################################################################################
# Compare finite-range fits with local Twiss derivatives
################################################################################

########################################
# First and second tune derivatives
########################################
local = line.twiss4d(chrom = True, delta_chrom = LOCAL_STEP)
xt.Table(
    data    = {
        'quantity': np.array(['dqx', 'ddqx', 'dqy', 'ddqy']),
        'scan_fit': np.array([
            fitted.dqx, fitted.ddqx, fitted.dqy, fitted.ddqy]),
        'local':    np.array([local.dqx, local.ddqx, local.dqy, local.ddqy])},
    index   = 'quantity').show(maxwidth = 'full')

########################################
# Second-order dispersion from the stored orbit samples
########################################
# Each scan point retains its Twiss table. ddx is d^2 x_co/d(delta)^2 [m],
# whereas a polynomial's quadratic orbit coefficient is half that derivative.
x_samples   = np.array([tw['x', OBSERVATION] for tw in fitted.twiss])
x_coeff     = np.polynomial.polynomial.polyfit(
    fitted.delta0, x_samples, deg = FIT_ORDER)
print('Second-order dispersion at', OBSERVATION, '[m]:')
print('Local Twiss ddx:', local['ddx', OBSERVATION])
print('Twice the quadratic orbit coefficient:', 2 * x_coeff[2])

################################################################################
# Test fit-window and polynomial-order sensitivity
################################################################################

########################################
# Repeat with a narrower range and a higher degree
########################################
narrow  = line.get_non_linear_chromaticity(
    delta0_range    = (-DELTA_MAX / 2, DELTA_MAX / 2),
    num_delta       = NUM_DELTA,
    fit_order       = FIT_ORDER,
    chrom           = False)
higher  = line.get_non_linear_chromaticity(
    delta0_range    = (-DELTA_MAX, DELTA_MAX),
    num_delta       = NUM_DELTA,
    fit_order       = FIT_ORDER + 1,
    chrom           = False)
print('Second tune derivatives: original / half range / higher degree')
print('x:', fitted.ddqx, narrow.ddqx, higher.ddqx)
print('y:', fitted.ddqy, narrow.ddqy, higher.ddqy)

########################################
# Compare tune predictions over their shared half-window
########################################
shared_delta = fitted.delta0[np.abs(fitted.delta0) <= DELTA_MAX / 2]
for label, alternative in [('Half range', narrow), ('Higher degree', higher)]:
    for plane in ['x', 'y']:
        derivatives     = getattr(alternative, f'dnq{plane}')
        coefficients    = derivatives / np.array([
            factorial(n) for n in range(len(derivatives))])
        base_coeff      = qx_coeff if plane == 'x' else qy_coeff
        change          = (
            np.polynomial.polynomial.polyval(shared_delta, coefficients)
            - np.polynomial.polynomial.polyval(shared_delta, base_coeff))
        print(label, plane, 'maximum tune-prediction change:',
              np.max(np.abs(change)))

# Derivative estimates depend on the fit window, degree and numerical precision;
# a higher polynomial degree alone does not establish convergence.
# In this window the tune predictions are more robust than their derivatives.

# The x-plane second derivative is particularly sensitive to local differencing;
# the printed comparisons are an error diagnostic, not an accuracy guarantee.

################################################################################
# Plot the tune curvature beyond the constant and linear terms
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for ax, sample, prediction, derivatives, plane in [
    (axes[0], fitted.qx, qx_fit, fitted.dnqx, 'x'),
    (axes[1], fitted.qy, qy_fit, fitted.dnqy, 'y'),
]:
    linear = derivatives[0] + derivatives[1] * fitted.delta0
    ax.plot(fitted.delta0 * 1e3, sample - linear, 'o', label = 'Twiss samples')
    ax.plot(fitted.delta0 * 1e3, prediction - linear, label = 'Polynomial fit')
    ax.set_ylabel(f'Q_{plane} minus linear terms')
    ax.legend()
axes[-1].set_xlabel('delta [1e-3]')
fig.tight_layout()
plt.show()
