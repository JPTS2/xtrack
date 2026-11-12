# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Fixed quadrupole, common particle batch and timing samples
########################################
LENGTH          = 1.2                    # [m]
K1              = 0.37                   # [m^-2]
N_PARTICLES     = 2048
N_REPEATS       = 5
SEED            = 12345
P0C             = 7e12                   # Proton reference momentum [eV/c]
N_SLICES        = [1, 2, 4, 8, 16, 32]
STAGES          = {'uniform': 1, 'yoshida-4': 3, 'yoshida-6': 7}

########################################
# Error scales and numerical reference
########################################
COORDINATES     = ['x', 'px', 'y', 'py', 'zeta']
SCALES          = np.array([1e-2, 1e-3, 1e-2, 1e-3, 1e-2])
REFERENCE_N     = 128                    # Yoshida-6 slices, then doubled
TARGET_ERROR    = 1e-7                   # Maximum dimensionless scaled error
# x,y,zeta scales are metres; px,py scales are relative to reference p0.

################################################################################
# Qualify the reference on the same finite-amplitude, off-momentum batch
################################################################################

########################################
# Use common initial conditions for every model and timing repetition
########################################
rng         = np.random.default_rng(SEED)
launch      = xt.Particles(
    'proton',
    p0c     = P0C,
    x       = rng.uniform(-1e-2, 1e-2, N_PARTICLES),
    px      = rng.uniform(-1e-3, 1e-3, N_PARTICLES),
    y       = rng.uniform(-1e-2, 1e-2, N_PARTICLES),
    py      = rng.uniform(-1e-3, 1e-3, N_PARTICLES),
    delta   = rng.uniform(-0.05, 0.05, N_PARTICLES),
)
magnet      = xt.Quadrupole(
    length      = LENGTH,
    k1          = K1,
    model       = 'drift-kick-drift-exact',
    integrator  = 'yoshida-6',
)
references  = []
for count in [REFERENCE_N, 2 * REFERENCE_N]:
    magnet.num_multipole_kicks  = 7 * count
    particles                   = launch.copy()
    magnet.track(particles)
    references.append(np.array([getattr(particles, name).copy()
                                for name in COORDINATES]))
reference_change    = np.max(
    np.abs(references[1] - references[0]) / SCALES[:, None])
print('CPU:', platform.processor(), 'Xtrack:', xt.__version__)
print('Reference refinement, max scaled error:', reference_change)
print('Particles:', N_PARTICLES, 'repeats:', N_REPEATS)

########################################
# Measure a warmed CPU tracking call, excluding setup and particle copies
########################################
records = []
for model in ['mat-kick-mat', 'mat-kick-mat-exact', 'drift-kick-drift-exact']:
    for integrator, stages in STAGES.items():
        for count in N_SLICES:
            candidate                       = magnet.copy()
            candidate.model                 = model
            candidate.integrator            = integrator
            candidate.num_multipole_kicks   = stages * count
            particles                       = launch.copy()
            candidate.track(particles)
            times   = []
            for repetition in range(N_REPEATS):
                particles   = launch.copy()
                started     = time.perf_counter()
                candidate.track(particles)
                times.append(time.perf_counter() - started)
            result  = np.array([getattr(particles, name)
                                for name in COORDINATES])
            error   = np.max(np.abs(result - references[1]) / SCALES[:, None])
            records.append((model, integrator, count, error,
                            np.median(times), np.min(times), np.max(times)))
            print(model, integrator, count, 'scaled error:', error,
                  'median/min/max [ms]:', np.array(records[-1][4:]) * 1e3)

################################################################################
# Compare cost at a common error bound, not at a common number of slices
################################################################################

########################################
# Select the cheapest measured configuration that reaches the requested accuracy
########################################
for model in ['mat-kick-mat', 'mat-kick-mat-exact', 'drift-kick-drift-exact']:
    eligible    = [row for row in records if row[0] == model
                and row[3] <= TARGET_ERROR]
    if reference_change > TARGET_ERROR / 10:
        print(model, 'reference needs further refinement for this target')
    elif eligible:
        best    = min(eligible, key = lambda row: row[4])
        print('Lowest measured cost at error <=', TARGET_ERROR, ':', best)
    else:
        print(model, 'no scanned configuration reaches the error target')

########################################
# Plot numerical error and approximation floors against measured cost
########################################
fig, axis   = plt.subplots(figsize = (9, 6), constrained_layout = True)
for model in ['mat-kick-mat', 'mat-kick-mat-exact', 'drift-kick-drift-exact']:
    for integrator in STAGES:
        rows    = [row for row in records if row[0:2] == (model, integrator)]
        axis.errorbar(
            [row[4] * 1e3 for row in rows], [row[3] for row in rows],
            xerr    = [[(row[4] - row[5]) * 1e3 for row in rows],
                       [(row[6] - row[4]) * 1e3 for row in rows]],
            fmt     = 'o-',
            label   = f'{model}, {integrator}',
        )
axis.set_xscale('log')
axis.set_yscale('log')
axis.axhline(TARGET_ERROR, color = 'k', linestyle = ':', label = 'Error target')
axis.set(xlabel = 'Median warmed tracking-call time [ms]',
         ylabel = 'Max scaled coordinate error')
axis.grid(which = 'both', alpha = 0.3)
axis.legend(fontsize = 8)
# mat-kick-mat uses an expanded kinetic model. More kicks cannot remove its
# finite-amplitude approximation error against an exact-drift reference.
# These timings include CPU call overhead; they are not isolated kernel times.
plt.show()
