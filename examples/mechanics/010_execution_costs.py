# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Measure setup, pose calculation, field writes and tracking separately.
# Compare turn-boundary and reference-speed passage sampling at high frequency.

################################################################################
# Required packages
################################################################################

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Lattice and execution context
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
CONTEXT      = xo.ContextCpu()                                    # Set ContextCupy on a machine with CUDA

########################################
# Support counts and tracking workload
########################################
SUPPORT_COUNTS = (1, 24, 168)
N_PARTICLES    = 128
N_TURNS        = 128

########################################
# Timing probes and projected history length
########################################
N_PROBES                = 5
PROJECTED_HISTORY_TURNS = 1_000_000   # Memory estimate only; not allocated

########################################
# Input motion for the within-turn phase test
########################################
AMPLITUDE_M    = 50e-9
FREQUENCY_HZ   = 20_000.              # Deliberately tests within-turn phase
INSPECT_TIME_S = .000123

################################################################################
# Set up the timing comparison
################################################################################

########################################
# Locate the demonstration lattice
########################################
repo_root = Path(__file__).resolve().parents[2]

########################################
# Prepare timing and comparison records
########################################
rows = []
timings = {}
phase_comparison = None

################################################################################
# Measure the same workload at each support count
################################################################################

########################################
# Repeat the complete measurement procedure
########################################
for count in SUPPORT_COUNTS:

    ########################################
    # Start timing the model load
    ########################################
    started = perf_counter()

    ########################################
    # Load independent study elements in the chosen context
    ########################################
    nominal_env = xt.load(repo_root / LATTICE_FILE)
    line = nominal_env['demo'].copy(_context = CONTEXT)
    env = line.env
    env['demo'] = line
    model_s = perf_counter() - started

    ########################################
    # Time the optional element selection separately
    ########################################
    started = perf_counter()

    ########################################
    # Select the quadrupoles for this workload
    ########################################
    mech = env.mechanics
    quads = list(line.get_table().rows.match(element_type = 'Quadrupole').name)[:count]

    ########################################
    # Record the selection cost
    ########################################
    selection_s = perf_counter() - started

    ########################################
    # Time explicit tracker construction
    ########################################
    started = perf_counter()
    line.build_tracker(_context = CONTEXT)
    build_s = perf_counter() - started

    ########################################
    # Start timing mechanics preparation
    ########################################
    started = perf_counter()

    ########################################
    # Create the support assignments
    ########################################
    girders = [line.mechanics.new_girder(f'g.{name}', elements = [name]) for name in quads]

    ########################################
    # Bind the prescribed high-frequency motion
    ########################################
    mech.add_source('tone', xt.mechanics.Harmonic(
        amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ,
    ), targets = [g.name for g in girders])

    ########################################
    # Record mechanics preparation cost
    ########################################
    setup_s = perf_counter() - started

    ########################################
    # Warm the kernel before timing steady tracking
    ########################################
    started = perf_counter()
    warm = line.build_particles(x = np.full(N_PARTICLES, 1e-7), y = np.full(N_PARTICLES, 1e-8))
    line.track(warm, num_turns = 1)
    CONTEXT.synchronize()
    warm_s = perf_counter() - started

    ########################################
    # Measure pure pose calculation
    ########################################
    started = perf_counter()
    for index in range(N_PROBES):
        mech.get_body_poses(time_s = INSPECT_TIME_S + index * 1e-6)
    pose_s = (perf_counter() - started) / N_PROBES

    ########################################
    # Measure numeric alignment application
    ########################################
    started = perf_counter()
    for index in range(N_PROBES):
        line.mechanics.apply(time_s = INSPECT_TIME_S + index * 1e-6)
    CONTEXT.synchronize()
    apply_s = (perf_counter() - started) / N_PROBES

    ########################################
    # Measure frozen tracking with the common particle workload
    ########################################
    particles_static = line.build_particles(x = np.full(N_PARTICLES, 1e-7), y = np.full(N_PARTICLES, 1e-8))
    started = perf_counter()
    line.track(particles_static, num_turns = N_TURNS)
    CONTEXT.synchronize()
    frozen_s = perf_counter() - started

    ########################################
    # Measure tracking with automatic motion updates
    ########################################
    particles_dynamic = line.build_particles(x = np.full(N_PARTICLES, 1e-7), y = np.full(N_PARTICLES, 1e-8))
    line.mechanics.enable_vibrations()
    started = perf_counter()
    line.track(particles_dynamic, num_turns = N_TURNS)
    CONTEXT.synchronize()
    dynamic_s = perf_counter() - started

    ########################################
    # Stop automatic replay and check particle survival
    ########################################
    line.mechanics.disable_vibrations()
    assert np.all(CONTEXT.nparray_from_context_array(particles_dynamic.state) > 0)

    ########################################
    # Prepare the turn-held comparison state
    ########################################
    line.mechanics.apply(time_s = INSPECT_TIME_S)
    held = line.mechanics.get_alignment_table().effective_shift_y.copy()

    ########################################
    # Measure reference-speed passage preparation
    ########################################
    started = perf_counter()
    line.mechanics.apply(time_s = INSPECT_TIME_S, sampling = 'passage')
    CONTEXT.synchronize()
    passage_s = perf_counter() - started
    passage = line.mechanics.get_alignment_table().effective_shift_y.copy()

    ########################################
    # Retain the spatial comparison and workload timings
    ########################################
    if count == max(SUPPORT_COUNTS):
        phase_comparison = (line.get_table()['s', quads], held, passage)
    rows.append(dict(count = count, model_s = model_s, selection_s = selection_s,
        setup_s = setup_s, build_s = build_s, warm_s = warm_s,
        pose_s = pose_s, apply_s = apply_s, frozen_s = frozen_s, dynamic_s = dynamic_s,
        passage_prepare_s = passage_s))
    for key, value in rows[-1].items():
        if key.endswith('_s'):
            timings[f'{count}_supports_{key}'] = value

################################################################################
# Interpret the measured costs and phase error
################################################################################

########################################
# Calculate the reference revolution time
########################################
turn_time = line.get_length() / (float(line.particle_ref.beta0[0]) * c)

########################################
# Compare setup, update and tracking costs
########################################
print('Reference backend execution costs\n----------------------------------')
print(f'Context: {type(CONTEXT).__name__}; particles: {N_PARTICLES}; turns: {N_TURNS}')
print('Supports   Load [s]   Select [ms]   Attach [s]   Build [s]   Warm [s]   Pose [ms]   Apply [ms]   Frozen [s]   Dynamic [s]')
for row in rows:
    print(f'{row["count"]:8d} {row["model_s"]:10.4f} {row["selection_s"] * 1e3:13.3f}'
          f' {row["setup_s"]:12.4f} {row["build_s"]:11.4f} {row["warm_s"]:10.4f}'
          f' {row["pose_s"] * 1e3:11.3f} {row["apply_s"] * 1e3:12.3f}'
          f' {row["frozen_s"]:12.4f} {row["dynamic_s"]:13.4f}')

########################################
# Quantify the phase missed by turn-held motion
########################################
print(f'Turn time [µs]: {turn_time * 1e6:.6f}; 2π f T_rev [rad]: {2 * np.pi * FREQUENCY_HZ * turn_time:.6f}')
print('Passage preparation [ms]:', ', '.join(f'{r["count"]}: {r["passage_prepare_s"] * 1e3:.3f}' for r in rows))
print(f'Largest held/passage alignment difference [nm]: {np.max(np.abs(held-passage)) * 1e9:.6f}')

########################################
# Estimate full-history memory without allocating it
########################################
history_bytes = PROJECTED_HISTORY_TURNS * max(SUPPORT_COUNTS) * 16 * 8
print(f'Unallocated full matrix history for {PROJECTED_HISTORY_TURNS:,} turns [GiB]: {history_bytes / 1024**3:.3f}')
print('The reference writes fields from Python. These measurements do not establish a fast GPU backend.')
print('Frozen/dynamic timings use the same particle workload, with different motion semantics.')

################################################################################
# Inspect cost scaling and passage-time effects
################################################################################

########################################
# Prepare runtime and spatial comparison panels
########################################
fig, axes = plt.subplots(1, 2, figsize = (10, 4))

########################################
# Plot measured per-turn and per-update costs
########################################
counts = [r['count'] for r in rows]
axes[0].plot(counts, [r['frozen_s'] / N_TURNS * 1e3 for r in rows], 'o-', label = 'Frozen tracking')
axes[0].plot(counts, [r['dynamic_s'] / N_TURNS * 1e3 for r in rows], 'o-', label = 'Host update + tracking')
axes[0].plot(counts, [r['apply_s'] * 1e3 for r in rows], 'o--', label = 'One explicit apply')
axes[0].set(xlabel = 'Supports', ylabel = 'Cost per turn/update [ms]', yscale = 'log')
axes[0].legend(fontsize = 8)

########################################
# Plot the difference between held and passage-time offsets
########################################
path_s, held, passage = phase_comparison
axes[1].plot(path_s, held * 1e9, '.-', label = 'Held at turn start')
axes[1].plot(path_s, passage * 1e9, '.-', label = 'Sampled at reference passage')

########################################
# Label and display the measured comparison
########################################
axes[1].set(xlabel = 'Nominal s [m]', ylabel = 'Effective y [nm]', title = f'{FREQUENCY_HZ:g} Hz coherent input')
axes[1].legend(fontsize = 8)
fig.tight_layout()
plt.show()
