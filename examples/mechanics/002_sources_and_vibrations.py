# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Enable physical-time vibration updates from harmonic, measured or PSD inputs.
# Compare one continuous run with the same run split into two tracking calls.
# Inspect how a rigid girder's rotation moves its two magnets differently.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from scipy.constants import c

################################################################################
# User parameters
################################################################################

########################################
# Choose the vertical and angular source models independently
########################################
SOURCE              = 'harmonic'       # 'harmonic', 'measured' or 'psd'
ROTATION_SOURCE     = 'harmonic'       # 'harmonic', 'measured' or 'psd'
INSTALLATION_Y_M    = .2e-6            # Static installation error, independent of dynamic seeds [m]

########################################
# Harmonic peaks and prescribed sample values
########################################
AMPLITUDE_M         = 50e-9            # Vertical peak for harmonic/measured inputs [m]
FREQUENCY_HZ        = 500.
PITCH_RAD           = 2e-7             # Survey pitch peak [rad]
YAW_RAD             = 3e-7             # Survey yaw peak [rad]
ROLL_RAD            = 4e-7             # Survey roll peak [rad]
DRIFT_M_PER_S       = 10e-9            # Independent slow longitudinal drift [m/s]

########################################
# One-sided PSD inputs and independent angular seeds
########################################
PSD_FREQUENCY_HZ    = [0., 100., 200., 800., 900.]
PSD_DENSITY_M2_HZ   = [0., 0., 1e-18, 1e-18, 0.]
PSD_SEED            = 956
# Angular PSD plateaux use peak**2 / 1400 rad²/Hz: the 700 Hz integrated
# trapezoidal band gives the same expected variance as the harmonic, peak²/2.

########################################
# Stored record and tracking duration
########################################
N_SAMPLES           = 4096
SAMPLE_STEP_S       = 1e-5              # 100 kHz record sampling; linear replay between samples
N_TURNS             = 8192
SPLIT_TURN          = N_TURNS // 2
PLOT_TURN_STRIDE    = 16               # Diagnostic queries only; tracking still updates every turn
INSPECT_TIME_S      = .0005            # Frozen optics time, independent of the tracking clock

################################################################################
# Build a periodic focusing model with one short girder
################################################################################

########################################
# Set up performance monitoring
########################################
timings = {}

########################################
# Construct two doublets separated by long drifts
########################################
# This is a periodic transport model with a straight survey. Example 004 uses
# a geometrically closed ring. The first doublet occupies a 1.5 m rigid girder.
started = perf_counter()
env = xt.Environment()
components = []
for index in range(4):
    env.new(f'q{index}', xt.Quadrupole, length = .5, k1 = .05 * (-1)**index)
    env.new(f'd{index}', xt.Drift, length = .5 if index % 2 == 0 else 100.)
    components += [f'q{index}', f'd{index}']
line = env.new_line(name = 'ring', components = components)
line.set_particle_ref('proton', p0c = 7e12)
timings['Line construction'] = perf_counter() - started

########################################
# Prepare the tracker and the common launch
########################################
started = perf_counter()
line.build_tracker()
launch = line.build_particles()
turn_period_s = line.get_length() / (float(launch.beta0[0]) * c)
duration_s = N_TURNS * turn_period_s
split_time_s = SPLIT_TURN * turn_period_s
timings['Tracker and launch preparation'] = perf_counter() - started

################################################################################
# Adopt the design placement and add the static installation error
################################################################################

########################################
# Attach q0 and q1 to one rigid girder
########################################
started = perf_counter()
mech = env.mechanics
line.mechanics.adopt_reference()
girder = line.mechanics.new_girder('g1', elements = ['q0', 'q1'])
girder.installation.shift_y = INSTALLATION_Y_M
timings['Reference and girder definition'] = perf_counter() - started

################################################################################
# Define vertical vibration, angular vibration and slow drift
################################################################################

########################################
# Choose the vertical displacement law
########################################
started = perf_counter()
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
if max(duration_s, INSPECT_TIME_S) > times[-1]:
    raise ValueError('Extend the stored record to cover tracking and frozen inspection')
if not 0 < SPLIT_TURN < N_TURNS:
    raise ValueError('Choose a split strictly inside the tracking run')
if SOURCE == 'harmonic':
    source = xt.mechanics.Harmonic(amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ)
elif SOURCE == 'measured':
    source = xt.mechanics.MeasuredMotion(
        time_s = [0., .002, .004, times[-1]], displacement = [0., AMPLITUDE_M, 0., 0.],
    )
elif SOURCE == 'psd':
    source = xt.mechanics.DisplacementPSD(
        frequency_hz = PSD_FREQUENCY_HZ, density = PSD_DENSITY_M2_HZ, seed = PSD_SEED,
    )
else:
    raise ValueError('Choose harmonic, measured or psd')
mech.add_source('vertical', source, targets = ['g1'], component = 'y')

########################################
# Bind pitch, yaw and roll in survey coordinates
########################################
# Positive survey pitch is a negative right-handed rotation about local x.
# These small angles compose as finite rotations in source insertion order.
for index, (component, amplitude) in enumerate([
    ('pitch', PITCH_RAD), ('yaw', YAW_RAD), ('roll', ROLL_RAD),
]):
    if ROTATION_SOURCE == 'harmonic':
        angular = xt.mechanics.Harmonic(
            amplitude = amplitude, frequency_hz = FREQUENCY_HZ, unit = 'rad',
        )
    elif ROTATION_SOURCE == 'measured':
        angular = xt.mechanics.MeasuredMotion(
            time_s = [0., .002, .004, times[-1]],
            displacement = [0., amplitude, 0., 0.], unit = 'rad',
        )
    elif ROTATION_SOURCE == 'psd':
        angular = xt.mechanics.AngularPSD(
            frequency_hz = PSD_FREQUENCY_HZ,
            density = np.array([0., 0., 1., 1., 0.]) * amplitude**2 / 1400.,
            seed = PSD_SEED + 1 + index,
        )
    else:
        raise ValueError('Choose harmonic, measured or psd angular motion')
    mech.add_source(component, angular, targets = ['g1'], component = component)

########################################
# Add a separate longitudinal drift
########################################
mech.add_source('slow_drift', xt.mechanics.MeasuredMotion(
    time_s = [times[0], times[-1]], displacement = [0., times[-1] * DRIFT_M_PER_S],
), targets = ['g1'], component = 's')
timings['Source definitions'] = perf_counter() - started

########################################
# Compute one record and reuse it for both tracking runs
########################################
# A PSD produces a seeded realization on this N/dt grid. Reusing the record
# keeps both runs on exactly the same realization, including across the split.
started = perf_counter()
motion = mech.compute_time_series(time_s = times)
timings['Motion record preparation'] = perf_counter() - started

########################################
# Inspect the source bindings and direct girder inputs separately
########################################
started = perf_counter()
source_table = mech.get_source_table()
table = motion.get_table('g1')
timings['Source and record inspection'] = perf_counter() - started

################################################################################
# Enable vibration updates and compare continuous and split tracking
################################################################################

########################################
# Track the whole interval in one call
########################################
# The default holds one mechanical pose for each complete turn, evaluated at
# that turn's entrance time. A new call continues from particles.time_s.
line.mechanics.enable_vibrations(time_series = motion)
continuous = launch.copy()
started = perf_counter()
line.track(continuous, num_turns = N_TURNS)
timings['Continuous tracking, first use'] = perf_counter() - started

########################################
# Track the same launch to the split point
########################################
particles = launch.copy()
started = perf_counter()
line.track(particles, num_turns = SPLIT_TURN)
timings['Split tracking, first call'] = perf_counter() - started
clock_at_split_s = float(particles.time_s)
q0_y_at_split = env['q0'].shift_y
q1_y_at_split = env['q1'].shift_y

########################################
# Continue the same particle without restarting the source phase
########################################
started = perf_counter()
line.track(particles, num_turns = N_TURNS - SPLIT_TURN)
timings['Split tracking, second call'] = perf_counter() - started
q0_y_at_end = env['q0'].shift_y
q1_y_at_end = env['q1'].shift_y

########################################
# Check all six final coordinates and the accumulated physical clock
########################################
for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
    np.testing.assert_array_equal(getattr(continuous, coordinate), getattr(particles, coordinate))
np.testing.assert_equal(continuous.time_s, particles.time_s)
np.testing.assert_equal(particles.at_turn[0], N_TURNS)
assert continuous.state[0] == particles.state[0] == 1

########################################
# Remove all dynamic inputs and retain the static installation error
########################################
line.mechanics.disable_vibrations()
np.testing.assert_equal(env['q0'].shift_y, INSTALLATION_Y_M)
np.testing.assert_equal(env['q1'].shift_y, INSTALLATION_Y_M)

################################################################################
# Inspect magnet poses at selected turn boundaries
################################################################################

########################################
# Query selected tracking times without building an every-turn pose history
########################################
# Include the last tracked turn of each call for the written-pose comparison.
# Thinning diagnostic queries does not change the motion applied during tracking.
started = perf_counter()
diagnostic_turns = np.unique(np.r_[
    np.arange(0, N_TURNS, PLOT_TURN_STRIDE), SPLIT_TURN - 1, N_TURNS - 1,
])
turn_times = diagnostic_turns * turn_period_s
elements = line.mechanics.get_table(time_s = INSPECT_TIME_S, time_series = motion)
q0_history = elements.rows['q0'].get_time_series(time_s = turn_times)
q1_history = elements.rows['q1'].get_time_series(time_s = turn_times)
timings['Read-only magnet histories'] = perf_counter() - started

########################################
# Compare actual written alignments with the requested entrance poses
########################################
# End-of-call fields retain the last tracked turn's pose, at (n-1)*T_rev.
# The particle clock has already advanced to n*T_rev at that point.
call_ends = np.searchsorted(diagnostic_turns, [SPLIT_TURN - 1, N_TURNS - 1])
for history, actual in [
    (q0_history, [q0_y_at_split, q0_y_at_end]),
    (q1_history, [q1_y_at_split, q1_y_at_end]),
]:
    np.testing.assert_allclose(
        actual, history.total_Y[call_ends], rtol = 0, atol = 1e-18,
    )

################################################################################
# Compare installed and frozen-vibration closed orbits
################################################################################

########################################
# Compute optics with only the static installation error
########################################
started = perf_counter()
static_twiss = line.twiss4d()

########################################
# Apply one fixed vibration time with automatic updates disabled
########################################
# Frozen Twiss repeats this one map. It is distinct from driven tracking.
mech.apply(time_s = INSPECT_TIME_S, time_series = motion)
frozen_twiss = line.twiss4d()
line.mechanics.disable_vibrations()
timings['Two frozen optics calculations'] = perf_counter() - started

################################################################################
# Explain the sources, clock continuity and rigid rotation
################################################################################

########################################
# Show the registered inputs and stored record boundaries
########################################
print('Prescribed girder motion\n------------------------')
print(f'Vertical source: {SOURCE}; angular sources: {ROTATION_SOURCE}.')
print(f'Static installation error y [µm]: {INSTALLATION_Y_M * 1e6:.3f}')
source_table.cols['name model component unit'].show()
print(f'Record: {N_SAMPLES} samples, dt = {SAMPLE_STEP_S * 1e6:g} µs, '
      f'0 <= t <= {times[-1] * 1e3:g} ms.')
print(f'Direct source-record payload [KiB]: {sum(v.nbytes for v in motion.values.values()) / 1024:.1f}')
print('Both runs replay the same record; tracking calls do not redraw a PSD realization.')

########################################
# Compare completed tracking times and final particle states
########################################
print('\nContinuous tracking and two consecutive calls\n'
      '--------------------------------------------')
print(f'Periodic model length [m]: {line.get_length():.3f}; revolution period [µs]: {turn_period_s * 1e6:.6f}')
print('One mechanical pose is held per turn, at that turn\'s entrance time.')
print(f'{"Tracking checkpoint":<31} {"Completed turns":>16} {"Particle time [ms]":>21}')
for label, turns, time_s in [
    ('One call: finished', N_TURNS, float(continuous.time_s)),
    ('Two calls: first call finished', SPLIT_TURN, clock_at_split_s),
    ('Two calls: second call finished', N_TURNS, float(particles.time_s)),
]:
    print(f'{label:<31} {turns:16d} {time_s * 1e3:21.9f}')
print('All six final particle coordinates and the physical clock agree exactly.')
print('The second call continues the waveform; it does not restart at t = 0.')

########################################
# Distinguish girder translation from motion at a magnet entrance
########################################
print('\nRigid girder rotation and magnet placement\n'
      '-----------------------------------------')
print('q0 and q1 share a 1.5 m girder; their entrance origins have different lever arms.')
print(f'Entrance motion at t = {INSPECT_TIME_S:g} s:')
for name in ('q0', 'q1'):
    lever_s = float(elements['reference_Z', name] - girder.nominal_frame.XYZ[2])
    print(f'{name}: entrance lever s = {lever_s:+.3f} m; '
          f'vertical vibration = {elements["dynamic_Y", name] * 1e9:+.3f} nm.')
print('To first order here, vertical entrance motion is girder y + pitch * lever s.')
print(f'After vibration and drift are removed, both magnets retain y = {INSTALLATION_Y_M * 1e6:.3f} µm.')

########################################
# Separate a frozen closed orbit from the driven trajectory
########################################
print('\nFrozen closed orbit at q0\n-------------------------')
print(f'Design + static installation error [µm]: {static_twiss.y[0] * 1e6:+.6f}')
print(f'With vibration applied at t = {INSPECT_TIME_S:g} s [µm]: {frozen_twiss.y[0] * 1e6:+.6f}')
print('These are closed orbits of two fixed maps, not the tracked particle coordinates.')

################################################################################
# Plot the inputs and magnet motion on explicit physical-time windows
################################################################################

########################################
# Show the vertical and angular inputs during the tracked interval
########################################
fig, axes = plt.subplots(2, 2, figsize = (11, 7))
show = times <= duration_s
axes[0, 0].plot(times[show] * 1e3, table.shift_y[show] * 1e9)
axes[0, 0].set(title = 'Girder: prescribed vertical vibration', ylabel = 'Vertical input [nm]')
for name, label in [('rot_x_rad', 'Pitch'), ('rot_y_rad', 'Yaw'), ('rot_s_rad', 'Roll')]:
    axes[0, 1].plot(times[show] * 1e3, table[name][show] * 1e6, label = label)
axes[0, 1].set(title = 'Girder: prescribed angular vibration', ylabel = 'Survey angle [µrad]')
axes[0, 1].margins(y = .25)
axes[0, 1].legend(loc = 'upper center', ncol = 3, fontsize = 9)

########################################
# Show both magnet entrances with equal line styles and sampling
########################################
for name, history in [('q0', q0_history), ('q1', q1_history)]:
    axes[1, 0].plot(turn_times * 1e3, history.dynamic_Y * 1e9, label = name)
axes[1, 0].set(title = 'Magnet entrances: translation + rotation', ylabel = 'Vertical vibration [nm]')
axes[1, 0].margins(y = .2)
axes[1, 0].legend(loc = 'upper center', ncol = 2)
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.axvline(split_time_s * 1e3, ls = ':', color = '0.35')
    ax.set_xlim(0., duration_s * 1e3)

########################################
# Show slow drift over the full record and mark the shorter tracked interval
########################################
axes[1, 1].plot(times * 1e3, table.shift_s * 1e9)
axes[1, 1].axvspan(0., duration_s * 1e3, alpha = .15, label = 'Tracked interval')
axes[1, 1].set(title = 'Girder: slow drift over the stored record', ylabel = 'Longitudinal drift [nm]')
axes[1, 1].legend(loc = 'upper left')
for ax in axes.flat:
    ax.set_xlabel('Physical time [ms]')
    ax.grid(alpha = .2)
fig.suptitle('Dotted lines mark the boundary between the two tracking calls', fontsize = 12)
fig.tight_layout()

################################################################################
# Review preparation, tracking and inspection costs separately
################################################################################

########################################
# Report first-use work separately from the two warm tracking calls
########################################
print('\nCPU operation costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<39} {seconds * 1e3:10.3f} ms')
print('The first tracking call includes mechanics preparation and possible compilation.')
print('Source records, diagnostic histories and frozen optics have separate costs.')

########################################
# Display the physical comparisons
########################################
plt.show()
