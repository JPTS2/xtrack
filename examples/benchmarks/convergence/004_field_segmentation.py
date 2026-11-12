# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicHermiteSpline
import xtrack as xt

################################################################################
# User parameters
################################################################################
LENGTH              = 1.0                   # Straight magnet length [m]
P0C                 = 1e9               # Proton reference momentum [eV/c]
COEFFICIENTS        = [0.05, 0.0, 0.0, 6.4, -19.2, 19.2, -6.4]
# By(x=0,y=0,s)/(B rho) = sum_j COEFFICIENTS[j] * s**j [m^-1].
Y_INITIAL           = 0.05                  # Off-axis probe [m]
SEGMENT_COUNTS      = [1, 2, 4, 8, 16, 32, 64, 128]
STEPS_PER_SEGMENT   = [8, 16]             # Integration refinement at each fit
REFERENCE_STEPS     = 1024
FIELD_SAMPLES       = 10                    # Interior samples per segment

################################################################################
# Retain the full smooth field as a separate physical reference
################################################################################
########################################
# Construct the complete polynomial and physical entrance state
########################################
profile = Polynomial(COEFFICIENTS)
exact   = xt.BFieldExpansion(
    length                  = LENGTH,
    knc                     = [COEFFICIENTS],
    num_phi                 = 7,
    num_integration_steps   = REFERENCE_STEPS,
    pkin_const              = True,
)
initial = xt.Particles(
    'proton', p0c = P0C, x = 0.001, px = 0.002,
    y   = Y_INITIAL, py = -0.001,
)
scales  = np.array([LENGTH, 1.0, LENGTH, 1.0, LENGTH, 1.0])[:, None]
# The exit error is dimensionless: x,y,zeta are divided by LENGTH; normalized
# kinetic momenta and delta retain their native dimensionless values.
# Entrance ax=ay=0 defines kinetic launch momenta. All exit comparisons below
# use kinetic momenta, including when interfaces retain canonical momenta.

########################################
# Refine native integration at fixed full-field representation
########################################
references  = []
for steps in [REFERENCE_STEPS // 2, REFERENCE_STEPS]:
    exact.num_integration_steps = steps
    particles                   = initial.copy()
    exact.track(particles)
    references.append(np.array([
        particles.x, particles.kin_px, particles.y, particles.kin_py,
        particles.zeta, particles.delta,
    ]))
reference           = references[-1]
reference_change    = np.max(np.abs((references[0] - reference) / scales))
print('Native reference refinement, scaled exit change:', reference_change)

################################################################################
# Fit cubic on-axis profiles and inspect their off-axis reconstructed fields
################################################################################

########################################
# Compare each local cubic with the original smooth field
########################################
errors  = {(mode, steps): [] for mode in [False, True]
                   for steps in STEPS_PER_SEGMENT}
field_errors    = []
for count in SEGMENT_COUNTS:
    edges   = np.linspace(0.0, LENGTH, count + 1)
    fit     = CubicHermiteSpline(
        edges, profile(edges), profile.deriv()(edges))
    segments        = []
    model_errors    = np.zeros(2)
    for index in range(count):
        length  = edges[index + 1] - edges[index]
        element = xt.BFieldExpansion(
            length                  = length,
            knc                     = [fit.c[:, index][::-1]],
            num_phi                 = 7,
            num_integration_steps   = 1,
        )
        segments.append(element)
        local_s = np.linspace(0.0, length, FIELD_SAMPLES + 2)[1:-1]
        for plane, y in enumerate([0.0, Y_INITIAL]):
            fitted  = element.get_field(x = 0.0, y = y, s_local = local_s)
            full    = exact.get_field(x = 0.0, y = y,
                                      s_local   = edges[index] + local_s)
            model_errors[plane] = max(
                model_errors[plane], np.max(np.abs(fitted['By'] - full['By'])))
    field_errors.append(model_errors)

    ########################################
    # Hold physical initial momenta fixed for both boundary conventions
    ########################################
    for mode in [False, True]:
        for steps in STEPS_PER_SEGMENT:
            particles   = initial.copy()
            if not mode:
                entrance    = segments[0].get_field(
                    x   = particles.x, y = particles.y, s_local = 0.0)
                particles.px   += entrance['Ax']
                particles.py   += entrance['Ay']
                particles.ax    = entrance['Ax']
                particles.ay    = entrance['Ay']
            # False preserves canonical momenta across fitted interfaces;
            # True preserves kinetic momenta instead. Vector-potential jumps
            # therefore matter even for nearly identical on-axis profiles.
            for element in segments:
                element.pkin_const              = mode
                element.num_integration_steps   = steps
                element.track(particles)
            values  = np.array([
                particles.x, particles.kin_px, particles.y, particles.kin_py,
                particles.zeta, particles.delta,
            ])
            error   = np.max(np.abs((values - reference) / scales))
            errors[mode, steps].append(error)
            print(f'segments={count:2d}, pkin_const={mode}, steps={steps}: '
                  f'scaled exit error={error:.6e}')
    print('Interior max |By_fit - By_full|/(B rho) [m^-1], on/off axis:',
          model_errors)

################################################################################
# Distinguish longitudinal fitting, off-axis fields and integration resolution
################################################################################
# Cubics omit fourth and higher longitudinal derivatives inside each segment.
# Their reconstructed off-axis field need not approach the full smooth field.
# num_phi=7 removes vertical truncation for each cubic, but cannot restore those
# missing derivatives. Increasing RK4 steps cannot repair that model mismatch.
# The field samples exclude interfaces; canonical transport additionally sees
# vector-potential jumps there. Finite-step RK4 is not exactly symplectic.
fig, axes   = plt.subplots(1, 2, figsize = (12, 5), layout = 'constrained')
for (mode, steps), values in errors.items():
    axes[0].loglog(SEGMENT_COUNTS, values, 'o-' if mode else 's--',
                   label    = f'pkin_const={mode}, RK4/segment={steps}')
axes[0].set(xlabel = 'Cubic segments',
            ylabel  = 'Maximum scaled exit-coordinate error')
for index, label in enumerate(['On axis', f'y={Y_INITIAL:g} m']):
    axes[1].loglog(SEGMENT_COUNTS, np.array(field_errors)[:, index],
                   'o-', label = label)
axes[1].set(xlabel = 'Cubic segments',
            ylabel  = r'Interior $|\Delta B_y/(B\rho)|$ [m$^{-1}$]')
for ax in axes:
    ax.grid(alpha = 0.25)
    ax.legend()
plt.show()
