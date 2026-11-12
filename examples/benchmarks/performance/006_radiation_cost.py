# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Model and execution context
########################################
LATTICE_FILE    = 'test_data/example_lattices/demo_ring/lattice.py'
BACKEND         = 'cpu'                  # cpu, openmp, cupy or opencl
CPU_THREADS     = 2                      # Used only for openmp

########################################
# Bounded tracking workload
########################################
N_PARTICLES     = 32
N_TURNS         = 4
N_REPEATS       = 5

########################################
# Common mean-radiation operating point
########################################
P0C             = 10e9                   # Electron reference p0*c [eV]
RF_VOLTAGE      = 50e6                   # [V]
HARMONIC        = 200

################################################################################
# Select and report the execution context
################################################################################

########################################
# Use the requested backend without an automatic fallback
########################################
if BACKEND == 'cpu':
    context = xo.ContextCpu(omp_num_threads = 0)
elif BACKEND == 'openmp':
    context = xo.ContextCpu(omp_num_threads = CPU_THREADS)
elif BACKEND == 'cupy':
    context = xo.ContextCupy()
elif BACKEND == 'opencl':
    context = xo.ContextPyopencl()
else:
    raise ValueError(f'Unknown backend: {BACKEND}')

########################################
# Record the software and workload
########################################
print('Host:', platform.platform(), platform.machine())
print('Xtrack / Xobjects:', xt.__version__, xo.__version__)
print('Context:', context)
if BACKEND in ['cpu', 'openmp']:
    threads = 1 if BACKEND == 'cpu' else CPU_THREADS
    print('CPU tracking threads:', threads)
print('Turns / repetitions:', N_TURNS, N_REPEATS)

################################################################################
# Prepare one compensated and tapered configuration
################################################################################

########################################
# Use the same geometry with scaled reference rigidity and native RF
########################################
repo_root           = Path(__file__).resolve().parents[3]
env                 = xt.load(repo_root / LATTICE_FILE)
line                = env['demo']
env['p0c']          = P0C
line.set_particle_ref('electron', p0c = P0C)
env['frequency_rf'] = 0.0
env.set('rf.1', harmonic = HARMONIC)
env['voltage_rf']   = RF_VOLTAGE
line.configure_radiation(model = 'mean')
line.compensate_radiation_energy_loss()
twiss               = line.twiss6d(chrom = False)
# Fixed normalized strengths scale physical fields with rigidity. This is the
# illustrative 10 GeV/c setting used in radiation/, not a hardware design.
# RF compensation and magnet taper are computed once and retained in all cases.

########################################
# Transfer to the requested context and initialize RNG streams publicly
########################################
line.discard_tracker()
line.build_tracker(_context = context)
initial = line.build_particles(
    particle_on_co  = twiss.particle_on_co,
    x               = np.linspace(-1e-5, 1e-5, N_PARTICLES),
    y               = 0.5e-5,
)
# A public draw initializes native streams without changing phase coordinates.
# Copies preserve these streams, so repeats use the same random realization.
# They measure timing spread, not independent Monte Carlo uncertainty.
rng = xt.RandomUniform(_context = context)
rng.generate(n_samples = N_PARTICLES, particles = initial)
context.synchronize()
print('Particles / elements:', N_PARTICLES, len(line.element_names))
print('Electron reference p0*c [GeV]:', P0C * 1e-9)
print('RF voltage [MV] / harmonic:', RF_VOLTAGE * 1e-6, HARMONIC)

################################################################################
# Measure only the change of radiation algorithm
################################################################################
medians     = []
lower       = []
upper       = []
survivors   = []
models      = [None, 'mean', 'quantum', 'quantum-kick']
for model in models:

    ########################################
    # Keep the same tapered fields, RF settings and entrance coordinates
    ########################################
    line.configure_radiation(model = model)
    warmup  = initial.copy()
    line.track(warmup, num_turns = N_TURNS)
    context.synchronize()
    # With radiation off the retained RF supplies unbalanced energy gain.
    # That case is a fixed-lattice cost baseline, not its own periodic solution.

    ########################################
    # Warm each model before repeated synchronized tracking-call timings
    ########################################
    samples = []
    alive   = []
    for repetition in range(N_REPEATS):
        particles   = initial.copy()
        context.synchronize()
        line.track(particles, num_turns = N_TURNS, time = True)
        samples.append(line.time_last_track)
        state       = context.nparray_from_context_array(particles.state)
        alive.append(np.count_nonzero(state > 0))
    medians.append(np.median(samples))
    lower.append(np.min(samples))
    upper.append(np.max(samples))
    survivors.append(np.min(alive))

################################################################################
# Report cost at this fixed field configuration
################################################################################

########################################
# Compare call time and launched particle-turn cost
########################################
labels      = np.array(['off', 'mean', 'quantum', 'quantum-kick'])
medians     = np.array(medians)
lower       = np.array(lower)
upper       = np.array(upper)
xt.Table(
    data    = {
        'model':            labels,
        'median_ms':        medians * 1e3,
        'min_ms':           lower * 1e3,
        'max_ms':           upper * 1e3,
        'us_launched_pt':   medians * 1e6 / (N_PARTICLES * N_TURNS),
        'min_survivors':    np.array(survivors),
    },
    index   = 'model',
).show(maxwidth = 'full')
# Loading, tapering, Twiss, RNG setup, copies, warmup and survival inspection
# are excluded. Timing includes the synchronized complete tracking call.
# The models have different stochastic physics; this is not an accuracy ranking.

########################################
# Plot medians with min-to-max timing spread
########################################
fig, ax = plt.subplots(figsize = (8, 4))
ax.errorbar(
    labels,
    medians * 1e3,
    yerr    = np.array([medians - lower, upper - medians]) * 1e3,
    fmt     = 'o',
)
ax.set_ylabel('Tracking-call time [ms]')
ax.set_ylim(bottom = 0)
ax.grid(axis = 'y')
fig.tight_layout()
plt.show()
