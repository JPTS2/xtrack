# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xtrack.tpsa as xtpsa

################################################################################
# User parameters
################################################################################

########################################
# Fixed nonlinear workload
########################################
MODEL_FILE  = 'test_data/example_lattices/demo_ring/lattice.py'
START       = 'arc.s1.c1.qf.in'
END         = 'arc.s1.c1.qf.out'
K2LF        = 1.0                       # Focusing sextupole k2*L [m^-2]
INITIAL     = dict(x = 1e-4, px = 2e-5, y = -2e-4, py = 3e-5,
                   zeta = 1e-3, delta = 2e-3)

########################################
# Scan and numerical agreement
########################################
ORDERS      = [1, 2, 3, 4]
REPEATS     = 5
WARMUPS     = 2
# Scales for (x, px, y, py, zeta, delta): metres and normalized momenta.
SCALES      = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
TOLERANCE   = 1e-9                      # Maximum scaled constant-part error

################################################################################
# Prepare one shared-model cell with the same scalar and map physics
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / MODEL_FILE)
env['k2lf'] = K2LF
line        = env['demo'].select(start = START, end = END)
# TPSA tracking is experimental, CPU only, and requires madng_tpsa.
# This is Xtrack tracking with GTPSA algebra, without a MAD-NG process.
line.config.XTRACK_MULTIPOLE_NO_SYNRAD  = True
line.build_tracker()
reference   = line.particle_ref
reference_args  = dict(
    p0c     = reference.p0c[0],
    mass0   = reference.mass0,
    q0      = reference.q0,
)
print('Xtrack version:', xt.__version__)
print('CPU serial; radiation off; no observation monitor')
print('Cell length [m] / elements:', line.get_length(), len(line.element_names))
print('Reference p0*c [GeV] / q0 [e]:', reference.p0c[0] * 1e-9,
      reference.q0)
print('Focusing sextupole k2*L [m^-2]:', K2LF)
print('Untimed warmups per case:', WARMUPS)
print('Initial coordinates (x, px, y, py, zeta, delta):', INITIAL)

################################################################################
# Separate fresh-input setup from warmed transport
################################################################################

########################################
# Construct a scalar probe or a six-coordinate identity map
########################################
def make_input(order):
    if order == 0:
        return xt.Particles(**reference_args, **INITIAL)
    return xtpsa.ParticlesTpsa(order = order, **reference_args, **INITIAL)

########################################
# Warm each case, then track each fresh input once
########################################
rows    = []
scalar  = None
for order in [0] + ORDERS:
    for repeat in range(WARMUPS):
        line.track(make_input(order))

    inputs      = []
    setup_times = []
    for repeat in range(REPEATS):
        before  = perf_counter()
        probe   = make_input(order)
        setup_times.append(perf_counter() - before)
        inputs.append(probe)

    track_times = []
    errors      = []
    for probe in inputs:
        before  = perf_counter()
        line.track(probe)
        track_times.append(perf_counter() - before)
        if order == 0:
            values  = np.array([getattr(probe, name)[0]
                               for name in INITIAL])
            scalar  = values
        else:
            values  = probe.const_part
        error   = np.max(np.abs((values - scalar) / SCALES))
        errors.append(error)
        if not np.isfinite(error) or error > TOLERANCE:
            raise RuntimeError('Scalar and TPSA constant parts disagree')

    rows.append((order, setup_times, track_times, max(errors)))

################################################################################
# Report cost and repeat spread
################################################################################
# Setup includes the input allocation and, for TPSA, descriptor/identity-map
# construction. Warmups and kernel compilation are excluded from both timings.
# Transport is one scalar trajectory versus one six-variable Taylor map;
# these perform different amounts of work. No speedup ratio is implied.
print('Times are median [min, max] over', REPEATS, 'fresh inputs [ms]')
for order, setup, tracking, error in rows:
    label   = 'scalar' if order == 0 else f'order {order}'
    print(f'{label:8s} setup {np.median(setup) * 1e3:.3f} '
          f'[{min(setup) * 1e3:.3f}, {max(setup) * 1e3:.3f}]; '
          f'track {np.median(tracking) * 1e3:.3f} '
          f'[{min(tracking) * 1e3:.3f}, {max(tracking) * 1e3:.3f}]; '
          f'max scaled orbit error {error:.3e}')

################################################################################
# Plot map-order scaling with the scalar baseline
################################################################################
fig, axes   = plt.subplots(1, 2, figsize = (10, 4), layout = 'constrained')
fig.suptitle(f'Native tracking: median and min/max of {REPEATS} fresh inputs')
labels      = ['scalar'] + [str(order) for order in ORDERS]
for axis, column, title in [(axes[0], 1, 'Fresh input setup'),
                            (axes[1], 2, 'Warmed cell transport')]:
    medians = np.array([np.median(row[column]) for row in rows]) * 1e3
    lower   = medians - np.array([min(row[column]) for row in rows]) * 1e3
    upper   = np.array([max(row[column]) for row in rows]) * 1e3 - medians
    axis.errorbar(labels, medians, yerr = [lower, upper], fmt = 'o',
                  capsize = 4)
    axis.set(xlabel = 'Scalar probe or TPSA map order', ylabel = 'Time [ms]',
             title = title, yscale = 'log')
    axis.grid(alpha = 0.3)
plt.show()
