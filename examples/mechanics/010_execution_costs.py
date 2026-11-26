# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# When is one rigid pose per turn an adequate approximation?
# Track the same launch through turn-held and reference-passage magnet positions,
# then separate their computational costs from loading, survey and compilation.

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
PRINT_TIMINGS  = False               # Detailed setup/update table; physical comparison is always shown

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
beam_comparison = None

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
    girders = line.mechanics.new_girders({f'g.{name}': [name] for name in quads})

    ########################################
    # Bind the prescribed high-frequency motion
    ########################################
    mech.add_source('tone', xt.mechanics.Harmonic(
        amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ,
    ), targets = list(girders))

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
    # Prepare the mechanics kernels before measuring the complete warm call.
    started = perf_counter()
    line.track(line.build_particles(x = np.full(N_PARTICLES, 1e-7), y = np.full(N_PARTICLES, 1e-8)), num_turns = 1)
    CONTEXT.synchronize()
    dynamic_prepare_s = perf_counter() - started
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
    if count == max(SUPPORT_COUNTS):
        probe_held = line.build_particles(x = 1e-7, y = 1e-8)
        line.track(probe_held, turn_by_turn_monitor = 'ONE_TURN_EBE')
        held_beam = CONTEXT.nparray_from_context_array(line.record_last_track.y)[0].copy()

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
        applied = line.mechanics.get_applied_table().rows[quads]
        passage_time_s = applied.pose_time_s.copy()
        phase_comparison = (line.get_table()['s', quads], held, passage)
        probe_passage = line.build_particles(x = 1e-7, y = 1e-8)
        line.track(probe_passage, turn_by_turn_monitor = 'ONE_TURN_EBE')
        passage_beam = CONTEXT.nparray_from_context_array(line.record_last_track.y)[0].copy()
        beam_comparison = (line.get_table().s, held_beam, passage_beam)
    rows.append(dict(count = count, model_s = model_s, selection_s = selection_s,
        setup_s = setup_s, build_s = build_s, warm_s = warm_s,
        pose_s = pose_s, apply_s = apply_s, frozen_s = frozen_s, dynamic_s = dynamic_s,
        passage_prepare_s = passage_s, dynamic_prepare_s = dynamic_prepare_s))
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
print('Turn-held approximation and reference-passage comparison\n'
      '-------------------------------------------------------')
print(f'Coherent vertical vibration: {AMPLITUDE_M * 1e9:g} nm peak at {FREQUENCY_HZ:g} Hz.')
print('All magnets move together at a given physical time; the beam reaches them at different times.')
print('Turn-held uses the entrance-time pose everywhere. Passage sampling uses t0 + s/(beta0*c).')
print('Both tracked probes below start at x = 100 nm, y = 10 nm, px = py = 0.')
path_s, held, passage = phase_comparison
beam_s, held_beam, passage_beam = beam_comparison
print(f'Observation at line end, held / passage y [nm]: {held_beam[-1] * 1e9:+.6f} / {passage_beam[-1] * 1e9:+.6f}')
print(f'Change caused by resolving the passage times [nm]: {(passage_beam[-1] - held_beam[-1]) * 1e9:+.6f}')
print('This is one-turn transport through two sampling approximations, with identical launch and magnetic strengths.')
print(f'Inspection time at line entrance [µs]: {INSPECT_TIME_S * 1e6:.6f}')
print(f'Applied magnet times [µs]: {passage_time_s.min() * 1e6:.6f} to {passage_time_s.max() * 1e6:.6f}')
print(f'Reference flight time for one turn [µs]: {turn_time * 1e6:.6f}')
print(f'Source phase advance during that passage [rad]: {2 * np.pi * FREQUENCY_HZ * turn_time:.6f}')
print(f'Largest applied-position difference between the approximations [nm]: {np.max(np.abs(held-passage)) * 1e9:.6f}')
print('Reference-passage sampling uses beta0: it omits particle-specific zeta/delta arrival delays.')
print('Each element still receives one rigid pose; neither comparison is continuous motion inside a thick element.')
print('Automatic updates are disabled. The final effective fields retain the displayed passage-sampled state.')

########################################
# Keep detailed timing and memory diagnostics optional
########################################
history_bytes = PROJECTED_HISTORY_TURNS * max(SUPPORT_COUNTS) * 16 * 8
if PRINT_TIMINGS:
    print('\nWarm tracking workload and setup costs\n--------------------------------------')
    print(f'Context: {type(CONTEXT).__name__}; particles: {N_PARTICLES}; turns: {N_TURNS}')
    print('Supports   Load [s]   Select [ms]   Attach [s]   Build [s]   Warm [s]   Pose [ms]   Apply [ms]   Frozen [s]   Dynamic [s]')
    for row in rows:
        print(f'{row["count"]:8d} {row["model_s"]:10.4f} {row["selection_s"] * 1e3:13.3f}'
              f' {row["setup_s"]:12.4f} {row["build_s"]:11.4f} {row["warm_s"]:10.4f}'
              f' {row["pose_s"] * 1e3:11.3f} {row["apply_s"] * 1e3:12.3f}'
              f' {row["frozen_s"]:12.4f} {row["dynamic_s"]:13.4f}')
    print('Reference-passage preparation [ms]:', ', '.join(f'{r["count"]}: {r["passage_prepare_s"] * 1e3:.3f}' for r in rows))
    print('Mechanics first-use preparation [ms]:', ', '.join(f'{r["count"]}: {r["dynamic_prepare_s"] * 1e3:.3f}' for r in rows))
    print(f'Unallocated full matrix history for {PROJECTED_HISTORY_TURNS:,} turns [GiB]: {history_bytes / 1024**3:.3f}')
    print('Explicit application probes use the host API; automatic tracking uses its selected prepared backend.')
    print('Warm tracking excludes first-use preparation. Actual GPU execution needs a device run.')
    print('Frozen/dynamic timings use the same workload but different motion semantics.')

################################################################################
# Show what the passage-time approximation changes physically
################################################################################

########################################
# Compare the magnet offsets encountered around the ring
########################################
fig, axes = plt.subplots(2, 1, figsize = (9, 6.5))
axes[0].plot(path_s, held * 1e9, label = 'Pose held at turn entrance')
axes[0].plot(path_s, passage * 1e9, label = 'Pose at reference-particle passage')
axes[0].set(xlabel = 'Quadrupole position s [m]', ylabel = 'Applied vertical displacement [nm]',
            title = f'One coherent {FREQUENCY_HZ:g} Hz vibration: entrance time {INSPECT_TIME_S * 1e6:g} µs')
axes[0].legend()

########################################
# Compare actual tracked beam trajectories through those two sets of poses
########################################
axes[1].plot(beam_s, held_beam * 1e9, label = 'Tracking with turn-held poses')
axes[1].plot(beam_s, passage_beam * 1e9, label = 'Tracking with passage-time poses')
axes[1].set(xlabel = 'Beam position s [m]', ylabel = 'Tracked vertical coordinate [nm]',
            title = 'Tracked one-pass beam: turn-held versus reference-speed passage poses')
axes[1].legend()
for ax in axes:
    ax.grid(alpha = .2)
fig.tight_layout()

################################################################################
# Review the cost of the two update operations and complete tracking calls
################################################################################

########################################
# Keep performance separate from the physical accuracy comparison
########################################
cost_fig, cost_ax = plt.subplots(figsize = (8, 4))
counts = [r['count'] for r in rows]
cost_ax.plot(counts, [r['frozen_s'] / N_TURNS * 1e3 for r in rows], 'o-', label = 'Frozen tracking, per turn')
cost_ax.plot(counts, [r['dynamic_s'] / N_TURNS * 1e3 for r in rows], 'o-', label = 'Automatic motion + tracking, per turn')
cost_ax.plot(counts, [r['apply_s'] * 1e3 for r in rows], 'o--', label = 'One explicit host application')
cost_ax.set(xlabel = 'Attached supports', ylabel = 'Warm cost [ms]', yscale = 'log',
            title = 'Warm tracking per turn and one host application\nLoading and first-use preparation excluded')
cost_ax.legend(fontsize = 9)
cost_ax.grid(alpha = .2)
cost_fig.tight_layout()
plt.show()
