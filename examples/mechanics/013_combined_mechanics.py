# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Assemble ground, feet, a local pump, response paths and installation on one ring.
# Connect the mounted source and response ports to a matched beam comparison.
# Compare an initial frozen mechanical map with the same map allowed to move.

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
# Lattice and execution context
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
CONTEXT      = xo.ContextCpu()                                    # ContextCupy on a machine with CUDA

########################################
# Independent installation and dynamic seeds
########################################
INSTALLATION_SEED = 956
GROUND_SEED       = 1956
REGION_SEED       = 1957

########################################
# Support contact geometry
########################################
FEET_LOCAL_M = [(-.3, -.5, -1.), (.3, -.5, -1.), (-.3, -.5, 1.), (.3, -.5, 1.)]

########################################
# Static installation scales
########################################
INSTALLATION_SIGMA_M   = (5e-6, 5e-6, 0.)
INSTALLATION_SIGMA_RAD = (0., 0., 1e-5)

########################################
# Local pump mount and excitation
########################################
PUMP_LOCAL_M = (.8, -.5, 0.)
PUMP_PEAK_M  = 30e-9
PUMP_HZ      = 25.

########################################
# Prescribed slow drift
########################################
DRIFT_M_PER_S = 1e-9

########################################
# Background and regional excess spectra
########################################
BACKGROUND_DENSITY = 1e-18            # One-sided vertical PSD [m²/Hz]
EXCESS_DENSITY     = 3e-18

########################################
# Spatial correlation and approximation controls
########################################
GROUND_LENGTH_M = 100.
REGION_WIDTH_M  = 80.
N_MODES         = 256

########################################
# Supplied conditional response parameters
########################################
GIRDER_HZ = 12.
MAGNET_HZ = 35.
BPM_HZ    = 8.
DAMPING   = .2

########################################
# Causal prehistory and motion sampling
########################################
SAMPLE_STEP_S = 1 / 1024
WARMUP_S      = 1.5                   # Source history starts at -WARMUP_S
DURATION_S    = .5

########################################
# Inspection and tracking interval
########################################
INSPECT_TIME_S = .25
N_TURNS        = 256
PRINT_TIMINGS  = False

########################################
# Explicit field preparation budget
########################################
MAX_FIELD_WORKSPACE_MIB = 256

################################################################################
# Set up the ordinary Xtrack line
################################################################################

timings = {}

started = perf_counter()

########################################
# Locate and load the demonstration ring
########################################
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy(_context = CONTEXT)
env = line.env
env['demo'] = line

timings['line_load_and_copy_s'] = perf_counter() - started

########################################
# Prepare and time the tracker before mechanics queries
########################################
started = perf_counter()
line.build_tracker(_context = CONTEXT)
timings['tracker_build_s'] = perf_counter() - started

################################################################################
# Define the nominal mechanical mounting hierarchy
################################################################################

started = perf_counter()

########################################
# Adopt the existing deliberate magnet geometry
########################################
line.mechanics.adopt_reference()

timings['reference_adoption_s'] = perf_counter() - started

########################################
# Access the shared mechanics model
########################################
mech = env.mechanics

started = perf_counter()

########################################
# Survey the reference ports and select magnets and pickups
########################################
table = line.get_table()
nominal = line.survey(include_element_frames = True)
quads = list(table.rows.match(element_type = 'Quadrupole').name)
bpms = list(table.rows['bpm[.].*'].name)

timings['geometry_selection_s'] = perf_counter() - started

started = perf_counter()

########################################
# Create one rigid support and magnet body per quadrupole
########################################
girders, magnets = [], []
for name in quads:
    frame = nominal.get_frame(name, which = 'ref_start')
    center = xt.Frame(frame.matrix).translate_s(float(env[name].length) / 2)
    girder = mech.new_girder(f'g.{name}', anchor = center, feet_local_m = FEET_LOCAL_M)
    magnet = mech.new_body(f'm.{name}', kind = 'magnet', parent = girder, anchor = frame, elements = [name], line = line)
    girders.append(girder)
    magnets.append(magnet)

########################################
# Mount each pickup on its nearest magnet body
########################################
pickups, pickup_magnets = [], []
for name in bpms:
    frame = nominal.get_frame(name, which = 'ref_start')
    nearest = int(np.argmin([np.linalg.norm(m.nominal_frame.XYZ - frame.XYZ) for m in magnets]))
    pickup_magnets.append(magnets[nearest])
    pickups.append(mech.new_body(f'p.{name}', kind = 'pickup', parent = magnets[nearest], anchor = frame,
                                 elements = [name], line = line))

timings['mechanics_definition_s'] = perf_counter() - started

################################################################################
# Define installation placement and mounted equipment
################################################################################

started = perf_counter()

########################################
# Draw one static installation case
########################################
installation = mech.generate_installation_errors(
    seed = INSTALLATION_SEED, bodies = [g.name for g in girders],
    translation_sigma_m = INSTALLATION_SIGMA_M, rotation_sigma_rad = INSTALLATION_SIGMA_RAD,
)

########################################
# Apply installation independently of dynamic source seeds
########################################
installation.apply(mech)

timings['installation_definition_s'] = perf_counter() - started

started = perf_counter()

########################################
# Attach the pump in its parent support's local frame
########################################
pump = mech.new_body('pump', kind = 'equipment', parent = girders[0], mount = xt.Frame.from_survey_angles(
    X = PUMP_LOCAL_M[0], Y = PUMP_LOCAL_M[1], Z = PUMP_LOCAL_M[2],
))

timings['mechanics_definition_s'] += perf_counter() - started

################################################################################
# Define the joint ground excitation
################################################################################

started = perf_counter()

########################################
# Include the specified source prehistory in the sample grid
########################################
times = np.arange(round((WARMUP_S + DURATION_S) / SAMPLE_STEP_S)) * SAMPLE_STEP_S - WARMUP_S

########################################
# Collect all stored feet in nominal world coordinates
########################################
positions = np.concatenate([g.nominal_feet_m for g in girders])

########################################
# Construct the shared background ground process
########################################
background = xt.mechanics.SpectralGroundMotion(
    [0., 1., 2., 30., 40.], [0., 0., BACKGROUND_DENSITY, BACKGROUND_DENSITY, 0.],
    seed = GROUND_SEED, correlation_length_m = GROUND_LENGTH_M, n_modes = N_MODES,
)

########################################
# Construct the independent regional excess process
########################################
regional = xt.mechanics.SpectralGroundMotion(
    [0., 1., 2., 30., 40.], [0., 0., EXCESS_DENSITY, EXCESS_DENSITY, 0.],
    seed = REGION_SEED, correlation_length_m = 30., n_modes = N_MODES,
)

########################################
# Combine the site background and regional envelope
########################################
site = xt.mechanics.RegionalGroundMotion(background, (
    xt.mechanics.GaussianRegion(regional, tuple(girders[0].nominal_frame.XYZ), REGION_WIDTH_M),
))

########################################
# Define the mounted pump's supplied floor propagation law
########################################
pump_source = xt.mechanics.LocatedSource(
    waveform = xt.mechanics.Harmonic(PUMP_PEAK_M, PUMP_HZ),
    position_m = tuple(pump.nominal_frame.XYZ), decay_length_m = 20., speed_m_s = 100.,
)

################################################################################
# Sample the floor and reconstruct support motion
################################################################################

########################################
# Sample the site field jointly at every contact
########################################
floor = xt.mechanics.sample_field(site, time_s = times, positions_m = positions,
                                 max_memory_bytes = MAX_FIELD_WORKSPACE_MIB * 1024**2)

########################################
# Add the coherent pump field at the same contacts
########################################
floor = floor + xt.mechanics.sample_field(pump_source, time_s = times, positions_m = positions,
                                         max_memory_bytes = MAX_FIELD_WORKSPACE_MIB * 1024**2)

########################################
# Fit six rigid coordinates and retain contact mismatch
########################################
fitted, residual = floor.fit_rigid_bodies(
    frames = [g.nominal_frame for g in girders], feet_local_m = np.array([g.feet_local_m for g in girders]),
    target_names = tuple(g.name for g in girders),
)

timings['floor_and_rigid_fit_s'] = perf_counter() - started

################################################################################
# Resolve the mechanical response paths
################################################################################

started = perf_counter()

########################################
# State the supplied response-loading assumptions
########################################
# These are supplied conditional diagonal responses at the stated ports, assumed
# to include their actual loading. They are not a derived multi-mass assembly;
# 006 shows that distinction and 012 accepts measured/coupled matrix responses.

########################################
# Apply the causal floor-to-support response
########################################
girder_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(GIRDER_HZ, DAMPING), sample_step_s = SAMPLE_STEP_S,
)
girder_motion = xt.mechanics.RigidMotion(times, girder_filter.process(fitted.values), fitted.target_names)

########################################
# Superpose a separate slow support drift
########################################
drift_values = np.zeros_like(girder_motion.values)
drift_values[..., 2] = np.maximum(times, 0.)[:, None] * DRIFT_M_PER_S
girder_motion = girder_motion.superpose(xt.mechanics.RigidMotion(times, drift_values, girder_motion.target_names))

########################################
# Transport each support input to its magnet port
########################################
magnet_input = girder_motion.transport(
    from_frames = [g.nominal_frame for g in girders], to_frames = [m.nominal_frame for m in magnets],
    target_names = tuple(m.name for m in magnets),
)

########################################
# Apply the conditional magnet response
########################################
magnet_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(MAGNET_HZ, DAMPING), sample_step_s = SAMPLE_STEP_S,
)
magnet_motion = xt.mechanics.RigidMotion(times, magnet_filter.process(magnet_input.values), magnet_input.target_names)

########################################
# Select the driving magnets and transport to pickup ports
########################################
pickup_input = magnet_motion.select([m.name for m in pickup_magnets],
    target_names = [p.name for p in pickups]).transport(
    from_frames = [m.nominal_frame for m in pickup_magnets], to_frames = [p.nominal_frame for p in pickups],
    target_names = tuple(p.name for p in pickups),
)

########################################
# Apply the conditional pickup response
########################################
pickup_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(BPM_HZ, DAMPING), sample_step_s = SAMPLE_STEP_S,
)
pickup_motion = xt.mechanics.RigidMotion(times, pickup_filter.process(pickup_input.values), pickup_input.target_names)

########################################
# Bind all resolved outputs as absolute body motion
########################################
for name, record in [('supports', girder_motion), ('magnets', magnet_motion), ('pickups', pickup_motion)]:
    mech.add_source(name, record, targets = record.target_names, reference = 'absolute')

timings['response_paths_s'] = perf_counter() - started

################################################################################
# Track the installed dynamic lattice
################################################################################

started = perf_counter()

########################################
# Apply the initial physical state
########################################
mech.apply(time_s = 0.)

########################################
# Launch on the installed closed orbit
########################################
launch = line.twiss4d().particle_on_co.copy()

timings['initial_frozen_twiss_s'] = perf_counter() - started

########################################
# Enable replay with an explicit physical origin
########################################
# First track the same launch with the initial mechanical pose held fixed.
frozen_beam = launch.copy()
line.track(frozen_beam, num_turns = N_TURNS, turn_by_turn_monitor = True)
frozen_y = CONTEXT.nparray_from_context_array(line.record_last_track.y)[0].copy()
line.mechanics.enable_vibrations(time_origin_s = 0.)

started = perf_counter()

########################################
# Advance the beam and check survival
########################################
line.track(launch, num_turns = N_TURNS, turn_by_turn_monitor = True)
moving_y = CONTEXT.nparray_from_context_array(line.record_last_track.y)[0].copy()
CONTEXT.synchronize()
assert np.all(CONTEXT.nparray_from_context_array(launch.state) > 0)
timings['dynamic_tracking_s'] = perf_counter() - started

########################################
# Disable replay while retaining installation placement
########################################
line.mechanics.disable_vibrations()

################################################################################
# Inspect the complete prescribed mechanical state
################################################################################

########################################
# Apply the selected inspection time
########################################
mech.apply(time_s = INSPECT_TIME_S)

########################################
# Read body poses and select the post-warm-up interval
########################################
survey = mech.survey(time_s = INSPECT_TIME_S)
positive = times >= 0

########################################
# Report geometry, response amplitudes and sampling accuracy
########################################
print('Combined prescribed mechanics on the demo ring\n----------------------------------------------')
print(f'Girders / magnets / pickups: {len(girders)} / {len(magnets)} / {len(pickups)}')
print(f'Feet: {len(positions)}; installation seed: {INSTALLATION_SEED}; dynamic seeds: {GROUND_SEED}, {REGION_SEED}')
print(f'Causal source prehistory [s]: {WARMUP_S:g}; step [s]: {SAMPLE_STEP_S:g}')
print(f'Linear-replay half-step amplitude error bound at 40 Hz [%]: {(1 - np.cos(np.pi * 40 * SAMPLE_STEP_S)) * 100:.4f}')
print(f'Rigid contact mismatch RMS after warm-up [nm]: {np.sqrt(np.mean(residual[positive]**2)) * 1e9:.6f}')
selected_pickup_index = 0
selected_magnet_index = magnets.index(pickup_magnets[selected_pickup_index])
selected_girder = girders[selected_magnet_index]
selected_magnet = magnets[selected_magnet_index]
selected_pickup = pickups[selected_pickup_index]
print(f'Selected physical path: {selected_girder.name} -> {selected_magnet.name} -> {selected_pickup.name}')
print('Stage         Selected-port y standard deviation after startup [nm]')

for label, record in [('Support', girder_motion), ('Magnet', magnet_motion), ('Pickup', pickup_motion)]:
    index = selected_pickup_index if label == 'Pickup' else selected_magnet_index
    print(f'{label:<13} {np.std(record.values[positive, index, 1]) * 1e9:30.6f}')
print(f'Tracked turns / physical duration [s]: {N_TURNS} / {launch.time_s:.9g}')
print(f'Installation y RMS unchanged [µm]: {np.std(installation.values[:, 1]) * 1e6:.6f}')
print(f'Largest moving-minus-initial-frozen beam y at ring origin [nm]: {np.max(np.abs(moving_y - frozen_y)) * 1e9:.6f}')
print('Both beam runs start from the same initial frozen closed orbit, with identical installation errors.')
print('The pickup response is a mechanical inspection port; no online feedback is applied.')
print('Sources use nominal reference geometry. The example does not infer force propagation or support loading.')

################################################################################
# Connect the physical source locations and mechanical response path
################################################################################

########################################
# Show where the common floor, regional excess and pump act on the ring
########################################
setup_fig, setup_axes = plt.subplots(1, 2, figsize = (11, 4.6), gridspec_kw = {'width_ratios': [1.1, 1]})
ring_xyz = np.column_stack((nominal.X, nominal.Y, nominal.Z))
setup_axes[0].plot(ring_xyz[:, 0], ring_xyz[:, 2], color = '.65', lw = 1)
anchors = np.array([g.nominal_frame.XYZ for g in girders])
floor_rms_nm = np.std(fitted.values[positive, :, 1], axis = 0) * 1e9
points = setup_axes[0].scatter(anchors[:, 0], anchors[:, 2], c = floor_rms_nm, s = 15, cmap = 'viridis')
setup_fig.colorbar(points, ax = setup_axes[0], label = 'Fitted floor y standard deviation [nm]', shrink = .75)
setup_axes[0].scatter(*pump.nominal_frame.XYZ[[0, 2]], marker = '*', s = 100, color = 'tab:red', label = 'Mounted pump')
setup_axes[0].scatter(*selected_girder.nominal_frame.XYZ[[0, 2]], marker = 'o', s = 100,
                       facecolors = 'none', edgecolors = 'black', label = 'Selected response path')
region = plt.Circle(girders[0].nominal_frame.XYZ[[0, 2]], REGION_WIDTH_M, fill = False,
                    ls = ':', color = 'tab:red', label = 'Regional width scale')
setup_axes[0].add_patch(region)
setup_axes[0].set(xlabel = 'World X [m]', ylabel = 'World Z [m]', title = 'One floor field sampled at every support')
setup_axes[0].set_aspect('equal', adjustable = 'datalim')
setup_axes[0].legend(fontsize = 8, loc = 'upper right')

########################################
# State which port and supplied loaded response each arrow represents
########################################
path_ax = setup_axes[1]
path_ax.set(xlim = (0., 1.), ylim = (0., 1.), title = 'From floor contacts to the selected pickup')
path_ax.axis('off')
for y, text in [
    (.89, 'Common ground + regional excess + pump field'),
    (.66, 'Four feet per support -> rigid fit\nContact mismatch retained; no girder bending'),
    (.43, f'Girder {GIRDER_HZ:g} Hz -> magnet {MAGNET_HZ:g} Hz'),
    (.20, f'Magnet -> pickup {BPM_HZ:g} Hz\nExplicit port transport before each response'),
]:
    path_ax.text(.5, y, text, ha = 'center', va = 'center', fontsize = 9,
                 bbox = {'boxstyle': 'round,pad=.55', 'fc': '#edf3f7', 'ec': '.55'})
for upper, lower in [(.82, .74), (.58, .50), (.35, .27)]:
    path_ax.annotate('', xy = (.5, lower), xytext = (.5, upper), arrowprops = {'arrowstyle': '->'})
path_ax.text(.5, .035, 'Responses are supplied conditional loaded responses.\nMounts alone do not determine the transfer functions.',
             ha = 'center', va = 'center', fontsize = 8)
setup_fig.tight_layout()

################################################################################
# Compare physical motion at a common pickup port and its beam consequence
################################################################################

########################################
# Transport the upstream records to the selected pickup reference frame
########################################
common_port_records = []
for label, record, index, origin in [
    ('Fitted floor', fitted, selected_magnet_index, selected_girder.nominal_frame),
    ('Girder response', girder_motion, selected_magnet_index, selected_girder.nominal_frame),
    ('Magnet response', magnet_motion, selected_magnet_index, selected_magnet.nominal_frame),
]:
    selected = record.select([record.target_names[index]], target_names = [selected_pickup.name])
    common_port_records.append((label, selected.transport(
        from_frames = [origin], to_frames = [selected_pickup.nominal_frame], target_names = [selected_pickup.name],
    )))
common_port_records.append(('Pickup response', pickup_motion.select([selected_pickup.name])))
fig, axes = plt.subplots(3, 1, figsize = (9, 8))
for label, record in common_port_records:
    axes[0].plot(times[positive], record.values[positive, 0, 1] * 1e9, label = label)
axes[0].set(ylabel = 'Port displacement [nm]', title = 'Same port and axes: compare the response stages')
axes[0].margins(y = .25)
axes[0].legend(fontsize = 8, ncol = 2, loc = 'upper center')

########################################
# Separate real relative motion from transport through a lever arm
########################################
axes[1].plot(times[positive], (magnet_motion.values[positive, selected_magnet_index, 1]
               - magnet_input.values[positive, selected_magnet_index, 1]) * 1e9,
               label = 'Magnet minus transported girder')
axes[1].plot(times[positive], (pickup_motion.values[positive, selected_pickup_index, 1]
               - pickup_input.values[positive, selected_pickup_index, 1]) * 1e9,
               label = 'Pickup minus transported magnet')
axes[1].set(xlabel = 'Physical time after tracking start [s]', ylabel = 'Relative displacement [nm]')
axes[1].margins(y = .25)
axes[1].legend(fontsize = 8, ncol = 2, loc = 'upper center')

########################################
# Show what changes when the same initial mechanical map moves during tracking
########################################
beam_times_ms = np.arange(N_TURNS) * line.get_length() / (float(line.particle_ref.beta0[0]) * 299792458.) * 1e3
axes[2].plot(beam_times_ms, (moving_y - frozen_y) * 1e9, color = 'tab:purple')
axes[2].set(xlabel = 'Physical tracking time [ms]', ylabel = 'Beam Δy [nm]',
            title = 'Actual tracked beam at the ring origin; identical launch and installation')
for ax in axes:
    ax.grid(alpha = .2)
fig.tight_layout()

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
if PRINT_TIMINGS:
    print('\nCPU operation costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<32} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
