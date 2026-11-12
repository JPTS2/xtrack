# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################
LENGTH              = 1.0                   # Straight magnet length [m]
P0C                 = 1e9               # Proton reference momentum [eV/c]
COEFFICIENTS        = [0.05, 0.0, 0.0, 6.4, -19.2, 19.2, -6.4]
# By(x=0,y=0,s)/(B rho) = sum_j COEFFICIENTS[j] * s**j [m^-1].
Y_INITIAL           = np.array([0.002, 0.01, 0.05])  # Off-axis probes [m]
STEP_COUNTS         = np.array([2, 4, 8, 16, 32, 64])
EXPANSION_ORDERS    = [3, 5, 7]             # Vertical scalar-potential order
REFERENCE_STEPS     = 1024

################################################################################
# Define one smooth polynomial field and physical entrance state
################################################################################
initial = xt.Particles(
    'proton',
    p0c = P0C,
    x   = np.full(len(Y_INITIAL), 0.001),
    px  = 0.002,
    y   = Y_INITIAL,
    py  = -0.001,
)
# ax=ay=0 initially, so px and py here are kinetic momenta divided by p0.
# pkin_const=True accepts this physical state and returns kinetic momenta.
# The straight degree-six dipole is fully represented at num_phi=7; reducing
# num_phi truncates its off-axis field even when on-axis profiles are identical.
scales  = np.array([LENGTH, 1.0, LENGTH, 1.0, LENGTH, 1.0])[:, None]

################################################################################
# Refine the native reference before comparing numerical resolutions
################################################################################
references  = []
for steps in [REFERENCE_STEPS // 2, REFERENCE_STEPS]:
    element = xt.BFieldExpansion(
        length                  = LENGTH,
        knc                     = [COEFFICIENTS],
        num_phi                 = 7,
        num_integration_steps   = steps,
        pkin_const              = True,
    )
    particles   = initial.copy()
    element.track(particles)
    values  = np.array([
        particles.x, particles.kin_px, particles.y, particles.kin_py,
        particles.zeta, particles.delta,
    ])
    references.append(values)
reference           = references[-1]
reference_change    = np.max(np.abs((references[1] - references[0]) / scales))
print('Reference refinement: maximum scaled exit change:', reference_change)
# Scale x,y,zeta by LENGTH and leave kin_px,kin_py,delta dimensionless. This is
# an absolute error norm, not coordinate-wise relative error near zero.

################################################################################
# Separate RK4 integration error from transverse-expansion error
################################################################################
errors      = []
scan_start  = perf_counter()
for order in EXPANSION_ORDERS:
    element = xt.BFieldExpansion(
        length                  = LENGTH,
        knc                     = [COEFFICIENTS],
        num_phi                 = order,
        num_integration_steps   = 1,
        pkin_const              = True,
    )
    order_errors    = []
    for steps in STEP_COUNTS:
        element.num_integration_steps   = int(steps)
        particles                       = initial.copy()
        element.track(particles)
        values  = np.array([
            particles.x, particles.kin_px, particles.y, particles.kin_py,
            particles.zeta, particles.delta,
        ])
        error   = np.max(np.abs((values - reference) / scales))
        order_errors.append(error)
        print(f'num_phi={order}, steps={steps:3d}: scaled error={error:.6e}')
    errors.append(order_errors)
print('Resolution scan wall time [s]:', perf_counter() - scan_start)
# RK4 is not exactly symplectic. This exit-coordinate comparison does not
# establish long-term stability or symplecticity.

################################################################################
# Display integration convergence and model-error floors
################################################################################
fig, ax = plt.subplots(figsize = (8, 5), layout = 'constrained')
for order, order_errors in zip(EXPANSION_ORDERS, errors):
    ax.loglog(STEP_COUNTS, order_errors, 'o-', label = f'num_phi={order}')
ax.axhline(reference_change, color = '0.4', ls = ':',
           label    = 'Reference refinement change')
ax.set(xlabel = 'RK4 steps through the complete magnet',
       ylabel   = 'Maximum scaled exit-coordinate error')
ax.grid(alpha = 0.25)
ax.legend()
plt.show()
