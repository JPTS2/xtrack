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
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################
LENGTH              = 0.8               # Straight magnet length [m]
P0C                 = 1e9               # Proton reference momentum [eV/c]
COEFFICIENTS        = [0.05, 0.02, 0.3, -0.4, 0.15]
# On-axis By/(B rho) = sum_j COEFFICIENTS[j] * s**j [m^-1].
TOLERANCE           = 1e-7              # Maximum scaled exit-coordinate error
MAX_STEPS           = 4096
N_PARTICLES         = 512               # Repeated calibration probes
REPEATS             = 5
Y_PROBES            = np.array([0.002, 0.01, 0.03, 0.05])  # [m]

################################################################################
# Represent the same quartic dipole in both native elements
################################################################################

########################################
# Convert normalized polynomial coefficients to SI spline boundary data
########################################
context = xo.ContextCpu(omp_num_threads = 0)
probes  = xt.Particles(
    'proton', _context = context, p0c = P0C,
    x   = 0.001, px = 0.002, y = Y_PROBES, py = -0.001,
)
rigidity    = probes.rigidity0[0]        # Signed reference B rho [T m]
profile     = Polynomial(COEFFICIENTS)
physical    = profile * rigidity
integral    = physical.integ()
spline      = xt.Spline4(
    val_start   = physical(0.0),
    der_start   = physical.deriv()(0.0),
    val_end     = physical(LENGTH),
    der_end     = physical.deriv()(LENGTH),
    mean        = (integral(LENGTH) - integral(0.0)) / LENGTH,
)
# Spline4 has five constraints, so it represents this quartic exactly.

########################################
# Keep the complete off-axis polynomial and compare its field components
########################################
elements    = {
    'BFieldExpansion': xt.BFieldExpansion(
        _context                = context,
        length                  = LENGTH,
        knc                     = [COEFFICIENTS],
        num_phi                 = 5,
        num_integration_steps   = 1,
        pkin_const              = True,
    ),
    'SplineBoris': xt.SplineBoris(
        _context    = context,
        length      = LENGTH,
        by          = spline,
        n_steps     = 1,
    ),
}
# ax=ay=0 at launch. BFieldExpansion's pkin_const=True accepts and returns
# kinetic momenta, matching SplineBoris's boundary convention.
ss, yy  = np.meshgrid(np.linspace(0.0, LENGTH, 17), Y_PROBES)
field   = elements['BFieldExpansion'].get_field(
    x   = 0.0, y = yy, s_local = ss)
expanded    = np.array([
    field[name] for name in ['Bx', 'By', 'Bs']]) * rigidity
boris_field         = np.array(elements['SplineBoris'].get_field(0.0, yy, ss))
field_difference    = np.max(np.abs(expanded - boris_field))
print('Maximum native field disagreement [T]:', field_difference)
if field_difference > 1e-12:
    raise RuntimeError('The two elements do not represent the same field.')

################################################################################
# Integrate the analytic kinetic equations with independent DOP853 refinement
################################################################################

########################################
# Define the field directly from the quartic polynomial
########################################
def kinetic_equations(s, flattened):
    x, px, y, py, zeta, delta   = flattened.reshape(6, len(Y_PROBES))
    ps                          = np.sqrt((1.0 + delta)**2 - px**2 - py**2)
    by                          = profile(s) - profile.deriv(2)(s) * y**2 / 2
    by += profile.deriv(4)(s) * y**4 / 24
    bs  = profile.deriv()(s) * y - profile.deriv(3)(s) * y**3 / 6
    return np.array([
        px / ps, py / ps * bs - by,
        py / ps, -px / ps * bs,
        1.0 - (1.0 + probes.beta0 * probes.ptau) / ps,
        np.zeros_like(delta),
    ]).ravel()


def exit_coordinates(particles):
    values  = np.array([
        particles.x, particles.kin_px, particles.y, particles.kin_py,
        particles.zeta, particles.delta,
    ])
    if np.any(particles.state <= 0) or not np.all(np.isfinite(values)):
        raise RuntimeError('Lost particles or non-finite tracking coordinates.')
    return values


########################################
# Resolve the independent reference more tightly than the accuracy target
########################################
scales      = np.array([LENGTH, 1.0, LENGTH, 1.0, LENGTH, 1.0])[:, None]
references  = []
for rtol, atol, subdivisions in [(2e-12, 2e-14, 16), (3e-14, 3e-16, 32)]:
    solution    = solve_ivp(
        kinetic_equations,
        (0.0, LENGTH),
        exit_coordinates(probes).ravel(),
        method      = 'DOP853',
        rtol        = rtol,
        atol        = atol,
        max_step    = LENGTH / subdivisions,
        t_eval      = [LENGTH],
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    references.append(solution.y[:, -1].reshape(6, len(Y_PROBES)))
reference           = references[-1]
reference_change    = np.max(np.abs((references[0] - reference) / scales))
print('DOP853 reference refinement, scaled exit change:', reference_change)
if reference_change > TOLERANCE / 100:
    raise RuntimeError('The reference is insufficiently resolved.')
# This is an absolute dimensionless norm: x,y,zeta are divided by LENGTH,
# while kinetic momenta/p0 and delta retain their native dimensionless values.

################################################################################
# Calibrate each integrator against the common error requirement
################################################################################
results         = {}
matched_target  = TOLERANCE
for name, element in elements.items():
    low, high   = 0, 1
    history     = {}
    while True:
        steps   = high if high - low <= 1 else (low + high) // 2
        if name == 'SplineBoris':
            element.n_steps = steps
        else:
            element.num_integration_steps   = steps
        particles   = probes.copy()
        element.track(particles)
        error   = np.max(np.abs(
            (exit_coordinates(particles) - reference) / scales))
        history[steps]  = error
        if error <= matched_target:
            high    = steps
            if high - low <= 1:
                break
        else:
            low = steps
            if high == steps:
                if high == MAX_STEPS:
                    raise RuntimeError(f'{name}: accuracy target not reached.')
                high    = min(2 * high, MAX_STEPS)
    results[name]   = dict(
        steps   = high, error = error, history = history, times = [])
    print(f'{name}: {high} steps, scaled exit error {error:.6e}')
    if name == 'BFieldExpansion':
        # Match Boris to the accuracy actually attained by the first integer
        # BFE step count, which can be better than the requested upper limit.
        matched_target  = error
print('Achieved-error target used for the timing comparison:', matched_target)
# The integer search assumes smooth convergence within the bracket. Actual
# achieved errors are reported; equal step counts would not mean equal accuracy.

################################################################################
# Time warmed serial CPU tracking with construction and copies excluded
################################################################################
bunch   = xt.Particles(
    'proton', _context = context, p0c = P0C,
    x   = 0.001, px = 0.002, y = np.resize(Y_PROBES, N_PARTICLES), py = -0.001,
)
bunch_reference = reference[:, np.arange(N_PARTICLES) % len(Y_PROBES)]
for name, element in elements.items():
    element.track(bunch.copy())
for repeat in range(REPEATS):
    # Alternate ordering to reduce timing-drift bias. CPU calls are synchronous.
    names   = list(elements) if repeat % 2 == 0 else list(reversed(elements))
    for name in names:
        particles   = bunch.copy()
        start       = perf_counter()
        elements[name].track(particles)
        elapsed = perf_counter() - start
        results[name]['times'].append(elapsed)
        error   = np.max(np.abs(
            (exit_coordinates(particles) - bunch_reference) / scales))
        if np.any(particles.state <= 0) or error > matched_target:
            raise RuntimeError(f'{name}: timed bunch failed the target.')
for name, result in results.items():
    times   = np.array(result['times']) * 1e3
    print(f'{name}: median {np.median(times):.4f} ms; '
          f'min/max {times.min():.4f}/{times.max():.4f} ms; '
          f'{N_PARTICLES} particles, {REPEATS} passes')
# This compares one field, launch ensemble and serial CPU. It establishes
# neither general hardware rankings nor long-term symplecticity.

################################################################################
# Display achieved accuracy and runtime at the common target
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (12, 5), layout = 'constrained')
for index, (name, result) in enumerate(results.items()):
    counts  = sorted(result['history'])
    axes[0].loglog(counts, [result['history'][n] for n in counts],
                   '.-', color = f'C{index}', label = name)
    axes[0].plot(result['steps'], result['error'], 'o', color = f'C{index}')
    times   = np.array(result['times']) * 1e3
    median  = np.median(times)
    axes[1].errorbar(index, median,
                     yerr   = [[median - times.min()], [times.max() - median]],
                     fmt    = 'o', capsize = 5, color = f'C{index}')
axes[0].axhline(matched_target, color = '0.4', ls = '--',
                label   = 'Matched achieved-error target')
axes[0].set(xlabel = 'Integrator steps',
            ylabel  = 'Scaled exit-coordinate error')
axes[0].legend()
axes[1].set(xticks = [0, 1], xticklabels = list(results),
            ylabel  = 'Tracking wall time [ms]', title = 'Median and min/max')
for ax in axes:
    ax.grid(alpha = 0.25)
plt.show()
