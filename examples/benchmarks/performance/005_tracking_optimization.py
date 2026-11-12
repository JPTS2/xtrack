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
N_PARTICLES     = 256
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
# Prepare equivalent source and optimized configurations
################################################################################

########################################
# Deep-copy the environment before disabling its expressions
########################################
repo_root   = Path(__file__).resolve().parents[3]
env         = xt.load(repo_root / LATTICE_FILE)
source      = env['demo']
optimized   = source.copy()
source.build_tracker(_context = context)
optimized.build_tracker(_context = context)
optimized.optimize_for_tracking()
# Optimization removes markers and inactive elements, replaces passive Devices
# with drifts and disables expressions in its owning environment. This copy is
# independent. The nominal aligned model has RF and radiation off.
print('Element counts (source / optimized):',
      len(source.element_names), len(optimized.element_names))
print('Circumferences [m]:', source.get_length(), optimized.get_length())
print('Electron reference p0*c [GeV]:', source.particle_ref.p0c[0] * 1e-9)
print('Particles:', N_PARTICLES)

########################################
# Keep a common physical launch at the same ring entrance
########################################
initial = source.build_particles(
    x       = np.linspace(-1e-4, 1e-4, N_PARTICLES),
    y       = 0.5e-4,
    delta   = 1e-4,
)

################################################################################
# Warm and measure each configuration
################################################################################
medians     = []
lower       = []
upper       = []
survivors   = []
outputs     = []
for line in [source, optimized]:

    ########################################
    # Compile and warm the complete workload outside the timer
    ########################################
    warmup  = initial.copy()
    line.track(warmup, num_turns = N_TURNS)
    context.synchronize()

    ########################################
    # Track a fresh identical batch for each timing repeat
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
    outputs.append(particles)

################################################################################
# Compare measured cost and physical output
################################################################################

########################################
# Report synchronized tracking-call cost and min-to-max spread
########################################
labels      = np.array(['source', 'optimized'])
medians     = np.array(medians)
lower       = np.array(lower)
upper       = np.array(upper)
xt.Table(
    data    = {
        'configuration':    labels,
        'median_ms':        medians * 1e3,
        'min_ms':           lower * 1e3,
        'max_ms':           upper * 1e3,
        'min_survivors':    np.array(survivors),
    },
    index   = 'configuration',
).show(maxwidth = 'full')
print('Measured source / optimized median time:', medians[0] / medians[1])
# Setup, optimization, compilation, copies and output inspection are excluded.
# This tracking-call comparison does not guarantee speedup on other hardware.

########################################
# Verify coordinates at the same physical entrance after complete turns
########################################
print('Coordinate  Maximum |optimized - source|  Unit')
for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
    before  = context.nparray_from_context_array(
        getattr(outputs[0], coordinate))
    after   = context.nparray_from_context_array(
        getattr(outputs[1], coordinate))
    unit    = 'm' if coordinate in ['x', 'y', 'zeta'] else '1'
    print(f'{coordinate:10s} {np.max(np.abs(after - before)):23.6e}  {unit}')
# Element indices and removed observation planes are not equivalent outputs.

########################################
# Show elapsed time without concealing repeat variability
########################################
fig, ax = plt.subplots(figsize = (7, 4))
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
