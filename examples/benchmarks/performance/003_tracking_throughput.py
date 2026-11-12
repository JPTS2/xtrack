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
PARTICLE_COUNTS = [1, 32, 256]
N_TURNS         = 8
N_REPEATS       = 5

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
# Load the fixed radiation-free ring
################################################################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
line        = env['demo']
line.build_tracker(_context = context)
print('Elements / circumference [m]:',
      len(line.element_names), line.get_length())
print('Electron reference p0*c [GeV]:', line.particle_ref.p0c[0] * 1e-9)
# This nominal 1 GeV/c ring has RF and radiation off. No monitor is recorded.

################################################################################
# Measure tracking calls at each particle count
################################################################################
medians     = []
lower       = []
upper       = []
survivors   = []
for count in PARTICLE_COUNTS:

    ########################################
    # Allocate deterministic probes and warm this workload
    ########################################
    initial = line.build_particles(
        x       = np.linspace(-1e-4, 1e-4, count),
        y       = 0.5e-4,
        delta   = 1e-4,
    )
    warmup  = initial.copy()
    line.track(warmup, num_turns = N_TURNS)
    context.synchronize()

    ########################################
    # Copy and synchronize before the native tracking-call timer
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
# Report cost and throughput with the measured spread
################################################################################

########################################
# Keep the full call distinct from a device-only kernel measurement
########################################
counts      = np.array(PARTICLE_COUNTS)
medians     = np.array(medians)
lower       = np.array(lower)
upper       = np.array(upper)
xt.Table(
    data    = {
        'particles':        counts,
        'median_ms':        medians * 1e3,
        'min_ms':           lower * 1e3,
        'max_ms':           upper * 1e3,
        'launched_pt_s':    counts * N_TURNS / medians,
        'min_survivors':    np.array(survivors),
    },
    index   = 'particles',
).show(maxwidth = 'full')
# time=True synchronizes the backend at the end of each tracking call. Setup,
# copies, warmup, observation and survival inspection are outside the timer.
# Launched particle-turns are a meaningful workload only alongside survival.

########################################
# Plot median cost per launched particle-turn and min-to-max spread
########################################
scale       = 1e6 / (counts * N_TURNS)
fig, ax     = plt.subplots(figsize = (8, 4))
ax.errorbar(
    counts,
    medians * scale,
    yerr    = np.array([medians - lower, upper - medians]) * scale,
    fmt     = 'o-',
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Launched particles')
ax.set_ylabel('Tracking-call cost [us / launched particle-turn]')
ax.grid()
fig.tight_layout()
plt.show()
