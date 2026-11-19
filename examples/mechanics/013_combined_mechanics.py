# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Assemble ground, feet, a local pump, response paths and installation on one ring.
# Track a short prescribed-motion interval after explicit response warm-up.

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Layout and independent static/dynamic seeds
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
CONTEXT = xo.ContextCpu()               # ContextCupy on a machine with CUDA
INSTALLATION_SEED = 956
GROUND_SEED       = 1956
REGION_SEED       = 1957
FEET_LOCAL_M = [(-.3, -.5, -1.), (.3, -.5, -1.), (-.3, -.5, 1.), (.3, -.5, 1.)]
INSTALLATION_SIGMA_M = (5e-6, 5e-6, 0.)
INSTALLATION_SIGMA_RAD = (0., 0., 1e-5)
PUMP_LOCAL_M = (.8, -.5, 0.)
PUMP_PEAK_M  = 30e-9
PUMP_HZ      = 25.
DRIFT_M_PER_S = 1e-9

########################################
# Floor and supplied conditional responses
########################################
BACKGROUND_DENSITY = 1e-18              # One-sided vertical PSD [m²/Hz]
EXCESS_DENSITY     = 3e-18
GROUND_LENGTH_M   = 100.
REGION_WIDTH_M    = 80.
N_MODES           = 256
GIRDER_HZ         = 12.
MAGNET_HZ         = 35.
BPM_HZ            = 8.
DAMPING           = .2

########################################
# Causal preparation and tracking
########################################
SAMPLE_STEP_S = 1 / 1024
WARMUP_S      = 1.5                    # Source history starts at -WARMUP_S
DURATION_S    = .5
INSPECT_TIME_S = .25
N_TURNS        = 32
MAX_FIELD_WORKSPACE_MIB = 256

################################################################################
# Separate supports, magnets and pickup bodies on the surveyed ring
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy(_context = CONTEXT)
env = line.env
env['demo'] = line
line.build_tracker(_context = CONTEXT)
line.mechanics.adopt_reference()
mech = env.mechanics
table = line.get_table()
nominal = line.survey(include_element_frames = True)
quads = list(table.rows.match(element_type = 'Quadrupole').name)
bpms = list(table.rows['bpm[.].*'].name)
girders, magnets = [], []
for name in quads:
    frame = nominal.get_frame(name, which = 'ref_start')
    center = xt.Frame(frame.matrix).translate_s(float(env[name].length) / 2)
    girder = mech.new_girder(f'g.{name}', anchor = center, feet_local_m = FEET_LOCAL_M)
    magnet = mech.new_body(f'm.{name}', parent = girder, anchor = frame, elements = [name], line = line)
    girders.append(girder)
    magnets.append(magnet)
pickups, pickup_magnets = [], []
for name in bpms:
    frame = nominal.get_frame(name, which = 'ref_start')
    nearest = int(np.argmin([np.linalg.norm(m.nominal_frame.XYZ - frame.XYZ) for m in magnets]))
    pickup_magnets.append(magnets[nearest])
    pickups.append(mech.new_body(f'p.{name}', parent = magnets[nearest], anchor = frame,
                                 elements = [name], line = line))
installation = mech.generate_installation_errors(
    seed = INSTALLATION_SEED, bodies = [g.name for g in girders],
    translation_sigma_m = INSTALLATION_SIGMA_M, rotation_sigma_rad = INSTALLATION_SIGMA_RAD,
)
installation.apply(mech)
pump = mech.new_body('pump', parent = girders[0], mount = xt.Frame.from_survey_angles(
    X = PUMP_LOCAL_M[0], Y = PUMP_LOCAL_M[1], Z = PUMP_LOCAL_M[2],
))
timings = {'layout_and_installation_s': perf_counter() - started}

################################################################################
# Jointly sample background, excess region and mounted pump at every foot
################################################################################
started = perf_counter()
times = np.arange(round((WARMUP_S + DURATION_S) / SAMPLE_STEP_S)) * SAMPLE_STEP_S - WARMUP_S
positions = np.concatenate([g.nominal_feet_m for g in girders])
background = xt.mechanics.SpectralGroundMotion(
    [0., 1., 2., 30., 40.], [0., 0., BACKGROUND_DENSITY, BACKGROUND_DENSITY, 0.],
    seed = GROUND_SEED, correlation_length_m = GROUND_LENGTH_M, n_modes = N_MODES,
)
regional = xt.mechanics.SpectralGroundMotion(
    [0., 1., 2., 30., 40.], [0., 0., EXCESS_DENSITY, EXCESS_DENSITY, 0.],
    seed = REGION_SEED, correlation_length_m = 30., n_modes = N_MODES,
)
site = xt.mechanics.RegionalGroundMotion(background, (
    xt.mechanics.GaussianRegion(regional, tuple(girders[0].nominal_frame.XYZ), REGION_WIDTH_M),
))
pump_source = xt.mechanics.LocatedSource(
    waveform = xt.mechanics.Harmonic(PUMP_PEAK_M, PUMP_HZ),
    position_m = tuple(pump.nominal_frame.XYZ), decay_length_m = 20., speed_m_s = 100.,
)
floor = xt.mechanics.sample_field(site, time_s = times, positions_m = positions,
                                 max_memory_bytes = MAX_FIELD_WORKSPACE_MIB * 1024**2)
floor = floor + xt.mechanics.sample_field(pump_source, time_s = times, positions_m = positions,
                                         max_memory_bytes = MAX_FIELD_WORKSPACE_MIB * 1024**2)
fitted, residual = floor.fit_rigid_bodies(
    frames = [g.nominal_frame for g in girders], feet_local_m = np.array([g.feet_local_m for g in girders]),
    target_names = tuple(g.name for g in girders),
)
timings['floor_and_rigid_fit_s'] = perf_counter() - started

################################################################################
# Resolve response paths before binding absolute motion to bodies
################################################################################
started = perf_counter()
# These are supplied conditional diagonal responses at the stated ports, assumed
# to include their actual loading. They are not a derived multi-mass assembly;
# 006 shows that distinction and 012 accepts measured/coupled matrix responses.
girder_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(GIRDER_HZ, DAMPING), sample_step_s = SAMPLE_STEP_S,
)
girder_motion = xt.mechanics.RigidMotion(times, girder_filter.process(fitted.values), fitted.target_names)
drift_values = np.zeros_like(girder_motion.values)
drift_values[..., 2] = np.maximum(times, 0.)[:, None] * DRIFT_M_PER_S
girder_motion = girder_motion.superpose(xt.mechanics.RigidMotion(times, drift_values, girder_motion.target_names))
magnet_input = girder_motion.transport(
    from_frames = [g.nominal_frame for g in girders], to_frames = [m.nominal_frame for m in magnets],
    target_names = tuple(m.name for m in magnets),
)
magnet_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(MAGNET_HZ, DAMPING), sample_step_s = SAMPLE_STEP_S,
)
magnet_motion = xt.mechanics.RigidMotion(times, magnet_filter.process(magnet_input.values), magnet_input.target_names)
pickup_input = magnet_motion.select([m.name for m in pickup_magnets],
    target_names = [p.name for p in pickups]).transport(
    from_frames = [m.nominal_frame for m in pickup_magnets], to_frames = [p.nominal_frame for p in pickups],
    target_names = tuple(p.name for p in pickups),
)
pickup_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(BPM_HZ, DAMPING), sample_step_s = SAMPLE_STEP_S,
)
pickup_motion = xt.mechanics.RigidMotion(times, pickup_filter.process(pickup_input.values), pickup_input.target_names)
for name, record in [('supports', girder_motion), ('magnets', magnet_motion), ('pickups', pickup_motion)]:
    mech.add_source(name, record, targets = record.target_names, reference = 'absolute')
timings['response_paths_s'] = perf_counter() - started

################################################################################
# Inspect a frozen state and track from its installed closed orbit
################################################################################
started = perf_counter()
mech.apply(time_s = 0.)
launch = line.twiss4d().particle_on_co.copy()
timings['initial_frozen_twiss_s'] = perf_counter() - started
line.mechanics.enable_vibrations(time_origin_s = 0.)
started = perf_counter()
line.track(launch, num_turns = N_TURNS)
CONTEXT.synchronize()
assert np.all(CONTEXT.nparray_from_context_array(launch.state) > 0)
timings['dynamic_tracking_s'] = perf_counter() - started
line.mechanics.disable_vibrations()
mech.apply(time_s = INSPECT_TIME_S)
survey = mech.survey(time_s = INSPECT_TIME_S)
positive = times >= 0
print('Combined prescribed mechanics on the demo ring\n----------------------------------------------')
print(f'Girders / magnets / pickups: {len(girders)} / {len(magnets)} / {len(pickups)}')
print(f'Feet: {len(positions)}; installation seed: {INSTALLATION_SEED}; dynamic seeds: {GROUND_SEED}, {REGION_SEED}')
print(f'Causal source prehistory [s]: {WARMUP_S:g}; step [s]: {SAMPLE_STEP_S:g}')
print(f'Linear-replay half-step amplitude error bound at 40 Hz [%]: {(1 - np.cos(np.pi * 40 * SAMPLE_STEP_S)) * 100:.4f}')
print(f'Rigid contact mismatch RMS after warm-up [nm]: {np.sqrt(np.mean(residual[positive]**2)) * 1e9:.6f}')
print('Stage         First-channel y RMS after warm-up [nm]')
for label, record in [('Support', girder_motion), ('Magnet', magnet_motion), ('Pickup', pickup_motion)]:
    print(f'{label:<13} {np.std(record.values[positive, 0, 1]) * 1e9:30.6f}')
print(f'Tracked turns / physical duration [s]: {N_TURNS} / {launch.time_s:.9g}')
print(f'Installation y RMS unchanged [µm]: {np.std(installation.values[:, 1]) * 1e6:.6f}')
print('Sources use nominal reference geometry. The example does not infer force propagation or support loading.')

################################################################################
# Compare the combined histories and surveyed ring motion
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for label, record in [('Floor fit', fitted), ('Support', girder_motion), ('Magnet', magnet_motion)]:
    axes[0].plot(times[positive], record.values[positive, 0, 1] * 1e9, label = label)
axes[1].plot(times[positive], girder_motion.values[positive, 0, 3] * 1e9, label = 'Support pitch')
axes[1].plot(times[positive], girder_motion.values[positive, 0, 5] * 1e9, label = 'Support roll')
axes[0].set(ylabel = 'First support path y [nm]', title = 'Combined sources and supplied causal responses')
axes[1].set(xlabel = 'Physical time after warm-up [s]', ylabel = 'Survey angle [nrad]', xlim = (0, min(1., DURATION_S)))
axes[0].legend(fontsize = 8)
axes[1].legend(fontsize = 8)
fig.tight_layout()
layout_fig, layout_axes = mech.plot(time_s = INSPECT_TIME_S, line = line,
    bodies = [g.name for g in girders], displacement_scale = 1e8, rotation_scale = 1e7, baseline = 'static')
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<32} {seconds:9.4f} s')
plt.show()
