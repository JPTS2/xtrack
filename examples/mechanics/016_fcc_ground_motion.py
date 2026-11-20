# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Put the requested FCC-ee development lattice on provisional quad supports.
# Resolve one common floor realization through girder and magnet responses.
# This is a runnable mechanics lesson, not an engineering support assignment.

################################################################################
# Required packages
################################################################################

import os
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
# FCC input and execution context
########################################
LATTICE_FILE = os.environ.get('XTRACK_MECHANICS_FCC_INPUT',
    '../inputs/acc-models-fcc-ee-108.1.0/lattices/z/fccee_z_beam_1_positrons.json')
LINE_NAME    = 'fccee_p_ring'
CONTEXT      = xo.ContextCpu()                  # ContextCupy on a CUDA machine

########################################
# Provisional support contacts and installation
########################################
FEET_LOCAL_M = [(-.3, -.5, -.5), (.3, -.5, -.5),
                (-.3, -.5, .5), (.3, -.5, .5)]
INSTALLATION_SEED    = 956
INSTALLATION_SIGMA_M = (.5e-6, .5e-6, 0.)        # Uncorrected demo seed, not an FCC tolerance

########################################
# One common ground PSD and correlation model
########################################
GROUND_FREQUENCY_HZ      = [0., 2., 3., 30., 40.]
GROUND_MODEL             = 'psd'                # 'psd' or 'plane_wave'; same response pipeline
GROUND_DENSITY_M2_PER_HZ = [0., 0., 1e-18, 1e-18, 0.]
GROUND_POLARIZATION     = (0., 1., 0.)           # World XYZ
GROUND_LENGTH_M         = 30.
GROUND_SEED             = 1956
N_MODES                 = 128                   # Convergence control, not a validated site model

########################################
# Optional plane wave in the same floor field
########################################
WAVE_PEAK_M       = 1e-9                        # Set to zero to omit
WAVE_HZ           = 25.
WAVE_DIRECTION    = (1., 0., 1.)                # World XYZ
WAVE_POLARIZATION = (.2, 1., .1)
WAVE_SPEED_M_S    = 200.

########################################
# Optional mounted pump and supplied propagation law
########################################
PUMP_PEAK_M         = 2e-9                      # Set to zero to omit
PUMP_HZ             = 55.
PUMP_LOCAL_M        = (.8, -.5, 0.)
PUMP_DECAY_LENGTH_M = 20.
PUMP_SPEED_M_S      = 100.

########################################
# Illustrative conditional response classes
########################################
GIRDER_RESONANCE_HZ = (8., 12., 18.)
GIRDER_DAMPING      = .3
MAGNET_RESONANCE_HZ = 80.
MAGNET_DAMPING      = .2
DRIFT_M_PER_S       = 1e-9                      # Prescribed local vertical drift after t=0

########################################
# One virtual pickup port downstream of the selected quad
########################################
PICKUP_AFTER_QUAD_M = .1
PICKUP_RESONANCE_HZ = 30.
PICKUP_DAMPING      = .3

########################################
# Bounded source record and causal startup
########################################
SAMPLE_STEP_S          = 1 / 1024
N_SAMPLES              = 512
PREHISTORY_S           = .4                     # Filters start at rest at this negative time
FIT_BLOCK_SAMPLES      = 64                     # Bound temporary arrays in the rigid contact fit
MAX_FIELD_WORKSPACE_MIB = 1024

########################################
# Tracking and repeated cost comparison
########################################
N_TURNS     = 128
PARTICLE_COUNTS = (1,)                          # Use (1, 32) for the longer throughput comparison
N_REPEATS   = 3

########################################
# Selected-port inspection and optional GIF
########################################
INSPECT_QUAD = None                             # None selects the first quadrupole
INSPECT_TIME_S = .02
DISPLACEMENT_SCALE = 2e7
ROTATION_SCALE     = 2e7
SAVE_GIF = False
GIF_FILE = 'fcc_quad_motion.gif'
PLOT_MAGNIFIED_SURVEY = True

################################################################################
# Load the existing FCC line and prepare ordinary tracking
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

########################################
# Check the finite record's sampling bandwidth
########################################
nyquist_hz = .5 / SAMPLE_STEP_S
active_frequencies_hz = [*GIRDER_RESONANCE_HZ, MAGNET_RESONANCE_HZ, PICKUP_RESONANCE_HZ]
if GROUND_MODEL == 'psd':
    active_frequencies_hz.append(max(GROUND_FREQUENCY_HZ))
if WAVE_PEAK_M:
    active_frequencies_hz.append(WAVE_HZ)
if PUMP_PEAK_M:
    active_frequencies_hz.append(PUMP_HZ)
if max(active_frequencies_hz) >= nyquist_hz:
    raise ValueError('Increase the record sample rate: an active source/response reaches Nyquist')
# Passing Nyquist is only necessary; decrease SAMPLE_STEP_S to assess phase and
# interpolation convergence before treating these responses as machine data.

########################################
# Locate the explicitly downloaded development lattice
########################################
repo_root = Path(__file__).resolve().parents[2]
lattice_path = (repo_root / LATTICE_FILE).resolve()
if not lattice_path.is_file():
    raise FileNotFoundError(
        f'{lattice_path}\nDownload the dev/108.1.0 FCC input described in review/FCC_INPUT.md.')

########################################
# Time lattice loading
########################################
started = perf_counter()
env = xt.load(lattice_path)
line = env[LINE_NAME]
timings['lattice_load_s'] = perf_counter() - started

########################################
# Time ordinary tracker preparation and one warm-up turn
########################################
started = perf_counter()
line.build_tracker(_context = CONTEXT)
line.track(line.build_particles(x = 1e-7, y = 1e-8), num_turns = 1)
CONTEXT.synchronize()
timings['tracker_and_first_nominal_turn_s'] = perf_counter() - started

########################################
# State the revolution clock and verify the requested interval
########################################
period_s = line.get_length() / (float(line.particle_ref.beta0[0]) * c)
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S - PREHISTORY_S
if times[0] > 0 or times[-1] < max(N_TURNS * period_s, INSPECT_TIME_S):
    raise ValueError('Extend the source record to contain tracking and inspection times')

################################################################################
# Measure the raw nominal lattice before attaching mechanics
################################################################################

########################################
# Repeat complete calls before any installation or vibration is applied
########################################
nominal_samples = {}
for count in PARTICLE_COUNTS:
    nominal_samples[count] = []
    for repeat in range(N_REPEATS):
        particles = line.build_particles(x = np.full(count, 1e-7), y = np.full(count, 1e-8))
        CONTEXT.synchronize()
        started = perf_counter()
        line.track(particles, num_turns = N_TURNS)
        CONTEXT.synchronize()
        nominal_samples[count].append(perf_counter() - started)
        assert np.all(CONTEXT.nparray_from_context_array(particles.state) > 0)
    timings[f'nominal_tracking_{count}p_median_s'] = float(np.median(nominal_samples[count]))

################################################################################
# Construct the provisional mechanical hierarchy
################################################################################

########################################
# Access mechanics and adopt deliberate offsets
########################################
mech = env.mechanics
started = perf_counter()
line.mechanics.adopt_reference()
timings['reference_adoption_s'] = perf_counter() - started

########################################
# Time required geometry selection and the reference-port survey
########################################
started = perf_counter()
table = line.get_table()
quads = list(table.rows.match(element_type = 'Quadrupole').name)
nominal = line.survey(include_element_frames = True)
selected_quad = quads[0] if INSPECT_QUAD is None else INSPECT_QUAD
if selected_quad not in quads:
    raise ValueError('INSPECT_QUAD must identify an existing thick quadrupole')
selected_index = quads.index(selected_quad)
timings['geometry_selection_and_survey_s'] = perf_counter() - started

########################################
# Time support definition
########################################
started = perf_counter()
girders, assignments = [], {}
class_indices = np.arange(len(quads)) % len(GIRDER_RESONANCE_HZ)

########################################
# Define one support per quadrupole with cyclic illustrative classes
########################################
# The catalogue does not establish engineering girders for this lattice.
# Replace this explicit assignment rule when verified support data are available.
for index, name in enumerate(quads):
    entry = nominal.get_frame(name, which = 'ref_start')
    center = xt.Frame(entry.matrix).translate_s(float(line[name].length) / 2)
    support_class = f'illustrative_{GIRDER_RESONANCE_HZ[class_indices[index]]:g}Hz'
    girder = mech.new_girder(f'g.{name}', anchor = center,
        feet_local_m = FEET_LOCAL_M, support_class = support_class)
    girders.append(girder)
    assignments[f'm.{name}'] = dict(elements = [name], parent = girder.name,
        anchor = entry, support_class = support_class)

########################################
# Attach all magnet children with one bulk survey
########################################
magnet_map = line.mechanics.new_bodies(assignments, kind = 'magnet')
magnets = list(magnet_map.values())
timings['mechanics_definition_s'] = perf_counter() - started

########################################
# Draw and apply installation independently of the ground seed
########################################
started = perf_counter()
installation = mech.generate_installation_errors(seed = INSTALLATION_SEED,
    bodies = [body.name for body in girders], translation_sigma_m = INSTALLATION_SIGMA_M)
installation.apply(mech)
mech.apply()
timings['installation_and_static_apply_s'] = perf_counter() - started

################################################################################
# Measure the frozen installed lattice before adding dynamic motion
################################################################################

########################################
# Keep every complete track call, synchronization and survival check visible
########################################
frozen_samples, frozen_final_x = {}, {}
for count in PARTICLE_COUNTS:
    frozen_samples[count] = []
    for repeat in range(N_REPEATS):
        particles = line.build_particles(x = np.full(count, 1e-7), y = np.full(count, 1e-8))
        CONTEXT.synchronize()
        started = perf_counter()
        line.track(particles, num_turns = N_TURNS)
        CONTEXT.synchronize()
        frozen_samples[count].append(perf_counter() - started)
        assert np.all(CONTEXT.nparray_from_context_array(particles.state) > 0)
        frozen_final_x[count] = CONTEXT.nparray_from_context_array(particles.x).copy()
    timings[f'frozen_tracking_{count}p_median_s'] = float(np.median(frozen_samples[count]))

################################################################################
# Sample one common floor realization at every support contact
################################################################################

########################################
# Start timing finite-record preparation
########################################
started = perf_counter()
positions = np.concatenate([body.nominal_feet_m for body in girders])

########################################
# Define the one-sided PSD and exponential amplitude coherency approximation
########################################
# Field receivers and pump positions use nominal geometry. Installation does
# not resample the prescribed small-signal field. GROUND_POLARIZATION applies
# only to scalar PSD input; a PlaneWave supplies its own vector polarization.
if GROUND_MODEL == 'psd':
    ground = xt.mechanics.SpectralGroundMotion(GROUND_FREQUENCY_HZ,
        GROUND_DENSITY_M2_PER_HZ, seed = GROUND_SEED,
        correlation_length_m = GROUND_LENGTH_M, n_modes = N_MODES)
elif GROUND_MODEL == 'plane_wave':
    ground = xt.mechanics.PlaneWave(WAVE_PEAK_M, WAVE_HZ, WAVE_DIRECTION,
        WAVE_SPEED_M_S, polarization = WAVE_POLARIZATION)
else:
    raise ValueError("GROUND_MODEL must be 'psd' or 'plane_wave'")
floor = xt.mechanics.sample_field(ground, time_s = times, positions_m = positions,
    polarization = GROUND_POLARIZATION, max_memory_bytes = MAX_FIELD_WORKSPACE_MIB * 1024**2)

########################################
# Add a directional plane wave before splitting the response classes
########################################
if WAVE_PEAK_M and GROUND_MODEL == 'psd':
    wave = xt.mechanics.PlaneWave(WAVE_PEAK_M, WAVE_HZ, WAVE_DIRECTION,
        WAVE_SPEED_M_S, polarization = WAVE_POLARIZATION)
    floor = floor + xt.mechanics.sample_field(wave, time_s = times,
        positions_m = positions, max_memory_bytes = MAX_FIELD_WORKSPACE_MIB * 1024**2)

########################################
# Mount a pump and supply its illustrative floor coupling
########################################
if PUMP_PEAK_M:
    pump = mech.new_body('pump', kind = 'equipment', parent = girders[selected_index],
        mount = xt.Frame.from_survey_angles(X = PUMP_LOCAL_M[0],
            Y = PUMP_LOCAL_M[1], Z = PUMP_LOCAL_M[2]))
    pump_source = xt.mechanics.LocatedSource(
        xt.mechanics.Harmonic(PUMP_PEAK_M, PUMP_HZ), tuple(pump.nominal_frame.XYZ),
        PUMP_DECAY_LENGTH_M, PUMP_SPEED_M_S)
    floor = floor + xt.mechanics.sample_field(pump_source, time_s = times,
        positions_m = positions, max_memory_bytes = MAX_FIELD_WORKSPACE_MIB * 1024**2)

########################################
# Fit rigid contacts in bounded time blocks and retain contact mismatch
########################################
if not isinstance(FIT_BLOCK_SAMPLES, (int, np.integer)) or FIT_BLOCK_SAMPLES < 2:
    raise ValueError('FIT_BLOCK_SAMPLES must be an integer of at least two')
support_frames = [body.nominal_frame for body in girders]
support_names = tuple(body.name for body in girders)
fitted_values = np.empty((len(times), len(girders), 6))
contact_residual_m = np.empty((len(times), len(girders)))
for start in range(0, len(times), FIT_BLOCK_SAMPLES):
    stop = min(start + FIT_BLOCK_SAMPLES, len(times))
    # FieldMotion requires two times. Overlap one sample for a final singleton;
    # each fit remains independent in time, with the same complete receiver set.
    first = min(start, stop - 2)
    block = xt.mechanics.FieldMotion(times[first:stop], positions, floor.values[first:stop])
    block_fit, block_residual_m = block.fit_rigid_bodies(
        frames = support_frames, feet_local_m = FEET_LOCAL_M, target_names = support_names)
    fitted_values[start:stop] = block_fit.values[start - first:]
    contact_residual_m[start:stop] = block_residual_m[start - first:]
fitted = xt.mechanics.RigidMotion(times, fitted_values, support_names)
del block, block_fit, block_residual_m, fitted_values
timings['common_floor_and_rigid_fit_s'] = perf_counter() - started

################################################################################
# Apply conditional girder and magnet transfer functions
################################################################################

########################################
# Start timing response preparation
########################################
started = perf_counter()
girder_values = np.empty_like(fitted.values)
girder_filters = []

########################################
# Feed each girder class from the same jointly sampled floor field
########################################
# These diagonal conditional responses are supplied assumptions, including their
# actual loading. They are not a derived coupled mechanical model or FCC data.
# Every filter starts at rest at times[0]; this is finite causal startup, not a
# claim that the last part of the record has reached stationary equilibrium.
for index, resonance_hz in enumerate(GIRDER_RESONANCE_HZ):
    selected = np.flatnonzero(class_indices == index)
    response = xt.mechanics.CausalResponse(
        oscillator = xt.mechanics.BaseExcitedOscillator(resonance_hz, GIRDER_DAMPING),
        sample_step_s = SAMPLE_STEP_S)
    girder_values[:, selected] = response.process(fitted.values[:, selected])
    girder_filters.append(response)

########################################
# Superpose a stated slow drift on the support displacement
########################################
girder_values[..., 1] += np.maximum(times, 0.)[:, None] * DRIFT_M_PER_S
girder_motion = xt.mechanics.RigidMotion(times, girder_values, fitted.target_names)

########################################
# Transport the moving support to each magnet's entrance port
########################################
magnet_input = girder_motion.transport(
    from_frames = [body.nominal_frame for body in girders],
    to_frames = [body.nominal_frame for body in magnets],
    target_names = tuple(body.name for body in magnets))

########################################
# Resolve the conditional girder-to-magnet response
########################################
magnet_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(MAGNET_RESONANCE_HZ, MAGNET_DAMPING),
    sample_step_s = SAMPLE_STEP_S)
magnet_motion = xt.mechanics.RigidMotion(times,
    magnet_filter.process(magnet_input.values), magnet_input.target_names)

########################################
# Mount one virtual pickup beyond the selected quad's exit
########################################
# This inspection port is not an identified FCC BPM or an online beam observer.
pickup = mech.new_body('virtual_pickup', kind = 'pickup',
    parent = magnets[selected_index], mount = xt.Frame.from_survey_angles(
        Z = float(line[selected_quad].length) + PICKUP_AFTER_QUAD_M))

########################################
# Transport the quad motion and apply the conditional pickup response
########################################
pickup_input = magnet_motion.select([magnets[selected_index].name],
    target_names = [pickup.name]).transport(
    from_frames = [magnets[selected_index].nominal_frame],
    to_frames = [pickup.nominal_frame], target_names = [pickup.name])
pickup_filter = xt.mechanics.CausalResponse(
    oscillator = xt.mechanics.BaseExcitedOscillator(PICKUP_RESONANCE_HZ, PICKUP_DAMPING),
    sample_step_s = SAMPLE_STEP_S)
pickup_motion = xt.mechanics.RigidMotion(times,
    pickup_filter.process(pickup_input.values), pickup_input.target_names)

########################################
# Bind resolved absolute outputs without adding inherited vibration twice
########################################
mech.add_source('resolved_girders', girder_motion,
    targets = girder_motion.target_names, reference = 'absolute')
mech.add_source('resolved_magnets', magnet_motion,
    targets = magnet_motion.target_names, reference = 'absolute')
mech.add_source('resolved_virtual_pickup', pickup_motion,
    targets = pickup_motion.target_names, reference = 'absolute')
timings['conditional_responses_and_bindings_s'] = perf_counter() - started

################################################################################
# Track the same installed FCC lattice with turn-held motion
################################################################################

########################################
# Time the first dynamic call separately from steady tracking
########################################
line.mechanics.enable_vibrations(time_origin_s = 0.)
started = perf_counter()
line.track(line.build_particles(x = 1e-7, y = 1e-8), num_turns = 1)
CONTEXT.synchronize()
timings['first_dynamic_track_1turn_s'] = perf_counter() - started

########################################
# Repeat complete dynamic calls on identical particle inputs
########################################
dynamic_samples, dynamic_final_x, slowdowns, nominal_slowdowns = {}, {}, {}, {}
for count in PARTICLE_COUNTS:
    dynamic_samples[count] = []
    for repeat in range(N_REPEATS):
        particles = line.build_particles(x = np.full(count, 1e-7), y = np.full(count, 1e-8))
        CONTEXT.synchronize()
        started = perf_counter()
        line.track(particles, num_turns = N_TURNS)
        CONTEXT.synchronize()
        dynamic_samples[count].append(perf_counter() - started)
        assert np.all(CONTEXT.nparray_from_context_array(particles.state) > 0)
        dynamic_final_x[count] = CONTEXT.nparray_from_context_array(particles.x).copy()
    timings[f'dynamic_tracking_{count}p_median_s'] = float(np.median(dynamic_samples[count]))
    slowdowns[count] = timings[f'dynamic_tracking_{count}p_median_s'] / timings[f'frozen_tracking_{count}p_median_s']
    nominal_slowdowns[count] = timings[f'dynamic_tracking_{count}p_median_s'] / timings[f'nominal_tracking_{count}p_median_s']

########################################
# Disable automatic motion and leave the selected inspection pose applied
########################################
line.mechanics.disable_vibrations()
mech.apply(time_s = INSPECT_TIME_S)

################################################################################
# Inspect the source inventory and selected mechanical ports
################################################################################

########################################
# Start optional diagnostic timing
########################################
started = perf_counter()
source_table = mech.get_source_table()
body_table = mech.get_table()
body_pose_table = mech.get_table(time_s = INSPECT_TIME_S)
element_table = line.mechanics.get_table(time_s = INSPECT_TIME_S)
quad_row = element_table.rows[selected_quad]
history_times = times[times >= 0.]
quad_history = quad_row.get_time_series(time_s = history_times)

########################################
# Report the model assumptions, bounded allocations and measured costs
########################################
print('FCC-ee prescribed ground motion\n-------------------------------')
print(f'Lattice: {lattice_path}')
print(f'Line / circumference [m]: {LINE_NAME} / {line.get_length():.6f}')
print(f'Quad supports / magnets / contacts: {len(girders)} / {len(magnets)} / {len(positions)}')
print('Assignment: one support per quad; cyclic response classes; NOT engineering girders.')
print('One virtual pickup port follows a conditional quad-to-pickup response; no online BPM/feedback coupling.')
print(f'Ground / installation seeds: {GROUND_SEED} / {INSTALLATION_SEED}')
print(f'Record samples / interval [s] / step [s]: {len(times)} / [{times[0]:g}, {times[-1]:g}] / {SAMPLE_STEP_S:g}')
print(f'Record sample rate / Nyquist [Hz]: {1 / SAMPLE_STEP_S:g} / {nyquist_hz:g}')
print('Causal filters start at rest at the beginning of this finite record.')
print(f'Floor record / bound response records [MiB]: {floor.nbytes / 1024**2:.2f} / {(girder_motion.nbytes + magnet_motion.nbytes + pickup_motion.nbytes) / 1024**2:.2f}')
print('The full records stay on the host; tracking uploads one current source block.')
print(f'Rigid contact mismatch RMS for t>=0 [nm]: {np.sqrt(np.mean(contact_residual_m[times >= 0.]**2)) * 1e9:.6f}')
print(f'Ground model: {GROUND_MODEL}; plane wave peak [nm]: {WAVE_PEAK_M * 1e9:g}; pump peak [nm]: {PUMP_PEAK_M * 1e9:g}')
print(f'Turns / particle counts / repeats / physical tracking duration [s]: {N_TURNS} / {PARTICLE_COUNTS} / {N_REPEATS} / {N_TURNS * period_s:.9g}')
print('Particles    Nominal [ms/turn]    Frozen [ms/turn]    Dynamic [ms/turn]    Moving/nominal    Moving/frozen')
print('Each timed packet has one motion phase; these are not packed bunches.')
for count in PARTICLE_COUNTS:
    nominal_ms = 1000 * timings[f'nominal_tracking_{count}p_median_s'] / N_TURNS
    frozen_ms = 1000 * timings[f'frozen_tracking_{count}p_median_s'] / N_TURNS
    dynamic_ms = 1000 * timings[f'dynamic_tracking_{count}p_median_s'] / N_TURNS
    print(f'{count:9d} {nominal_ms:20.6f} {frozen_ms:19.6f} {dynamic_ms:20.6f} {nominal_slowdowns[count]:17.4f} {slowdowns[count]:16.4f}')
    print(f'  First-particle final x change from frozen case [nm]: {(dynamic_final_x[count][0] - frozen_final_x[count][0]) * 1e9:.6f}')
source_table.cols['name model reference n_targets'].show()

################################################################################
# Plot the floor-to-girder-to-magnet hierarchy
################################################################################

########################################
# Compare records at the selected support and magnet ports
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (9, 6))
for label, record in [('Floor fit at support', fitted),
                       ('Girder at support', girder_motion)]:
    axes[0].plot(times, record.values[:, selected_index, 1] * 1e9, label = label)
axes[0].plot(times, magnet_motion.values[:, selected_index, 1] * 1e9,
             label = 'Magnet at entrance', alpha = .85)
axes[0].plot(times, pickup_motion.values[:, 0, 1] * 1e9,
             label = 'Virtual pickup beyond quad', alpha = .85)
axes[0].axvline(0., color = '.6', ls = '--')
axes[0].set(ylabel = 'Vertical port motion [nm]', title = selected_quad)
relative_y = magnet_motion.values[:, selected_index, 1] - magnet_input.values[:, selected_index, 1]
axes[1].plot(times, relative_y * 1e9, label = 'Magnet minus transported girder')
axes[1].plot(times, (pickup_motion.values[:, 0, 1] - pickup_input.values[:, 0, 1]) * 1e9,
             label = 'Virtual pickup minus transported quad')
axes[1].axvline(0., color = '.6', ls = '--', label = 'Tracking starts')
axes[1].set(xlabel = 'Physical time [s]', ylabel = 'Relative vertical motion [nm]')
for axis in axes:
    axis.legend()
    axis.grid(alpha = .2)
fig.tight_layout()

########################################
# Plot the selected quad's composed six-coordinate history
########################################
quad_history.plot()

########################################
# Compare the prescribed floor PSD and supplied girder responses
########################################
fig = plt.figure(figsize = (10, 7))
layout = fig.add_gridspec(2, 2)
psd_axis = fig.add_subplot(layout[0, 0])
response_axis = fig.add_subplot(layout[0, 1])
ring_axis = fig.add_subplot(layout[1, :])
if GROUND_MODEL == 'psd':
    psd_axis.plot(GROUND_FREQUENCY_HZ, GROUND_DENSITY_M2_PER_HZ)
    psd_axis.set(xlabel = 'Frequency [Hz]', ylabel = 'One-sided PSD [m²/Hz]',
                title = 'Supplied stochastic ground spectrum')
else:
    psd_axis.stem([WAVE_HZ], [WAVE_PEAK_M * 1e9])
    psd_axis.set(xlabel = 'Frequency [Hz]', ylabel = 'Wave peak displacement [nm]',
                title = 'Stated plane-wave input')
response_frequencies = np.linspace(.1, 100., 600)
for resonance_hz in GIRDER_RESONANCE_HZ:
    response = xt.mechanics.BaseExcitedOscillator(resonance_hz, GIRDER_DAMPING)
    response_axis.plot(response_frequencies, np.abs(response.evaluate(response_frequencies)),
                 label = f'Girder {resonance_hz:g} Hz')
response_axis.set(xlabel = 'Frequency [Hz]', ylabel = 'Displacement / base displacement',
            title = 'Illustrative conditional response classes')
response_axis.legend()
quad_s_km = np.array([table['s', name] for name in quads]) / 1000
girder_y_nm = np.array([body_pose_table['dynamic_Y', body.name] for body in girders]) * 1e9
ring_axis.plot(quad_s_km, girder_y_nm, label = 'Girders', lw = 1.)
ring_axis.plot(quad_s_km, element_table.dynamic_Y * 1e9,
               label = 'Quadrupoles', alpha = .7, lw = .8)
ring_axis.set(xlabel = 'Ring coordinate s [km]', ylabel = 'Dynamic world Y [nm]',
              title = f'Whole FCC ring at t={INSPECT_TIME_S:g} s')
ring_axis.legend()
for axis in (psd_axis, response_axis, ring_axis):
    axis.grid(alpha = .2)
fig.tight_layout()

########################################
# Show selected support feet and magnet span at magnified scale
########################################
if PLOT_MAGNIFIED_SURVEY:
    mech.plot(time_s = INSPECT_TIME_S, bodies = [girders[selected_index].name,
        magnets[selected_index].name], line = line, view = 'bodies', baseline = 'static',
        displacement_scale = DISPLACEMENT_SCALE, rotation_scale = ROTATION_SCALE)

########################################
# Optionally animate the selected quad with magnified rigid motion
########################################
if SAVE_GIF:
    animation = quad_row.animate(time_s = np.linspace(0., N_TURNS * period_s, 48),
        baseline = 'static', displacement_scale = DISPLACEMENT_SCALE,
        rotation_scale = ROTATION_SCALE)
    animation.save(GIF_FILE, writer = 'pillow', fps = 24)

timings['optional_tables_plots_and_gif_s'] = perf_counter() - started
print('\nSeparate operation costs [s]')
for name, value in timings.items():
    print(f'{name:<43} {value:10.6f}')
print(f'Nominal tracking repeat costs [s]: {nominal_samples}')
print(f'Frozen tracking repeat costs [s]: {frozen_samples}')
print(f'Dynamic tracking repeat costs [s]: {dynamic_samples}')
print('Automatic vibrations disabled; inspect mech, source_table, body_table, element_table and quad_history.')
plt.show()
