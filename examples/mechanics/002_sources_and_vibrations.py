# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Compare harmonic, measured and PSD inputs on a six-coordinate rigid support.
# Track a reusable record, stop vibration and retain the installation placement.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Installation, source and slow drift
########################################
SOURCE          = 'harmonic'            # 'harmonic', 'measured' or 'psd'
ROTATION_SOURCE = 'harmonic'            # 'harmonic', 'measured' or 'psd'
INSTALLATION_Y  = 20e-6                 # [m], fixed across dynamic seeds
AMPLITUDE_M     = 50e-9                 # Peak harmonic displacement [m]
FREQUENCY_HZ    = 500.
PSD_SEED        = 956
PITCH_RAD       = 2e-7                  # Peak survey pitch [rad]
YAW_RAD         = 3e-7
ROLL_RAD        = 4e-7
DRIFT_M_PER_S   = 1e-9                  # Slow prescribed drift, not stochastic vibration

########################################
# Sampling and tracking
########################################
N_SAMPLES       = 4096
SAMPLE_STEP_S   = 1e-4
N_TURNS         = 256
INSPECT_TIME_S  = .0005

################################################################################
# Prepare an ordinary periodic focusing line and its installation
################################################################################
started = perf_counter()
env = xt.Environment()
components = []
for index in range(4):
    env.new(f'q{index}', xt.Quadrupole, length = 1., k1 = .005 * (-1)**index)
    env.new(f'd{index}', xt.Drift, length = 100.)
    components += [f'q{index}', f'd{index}']
line = env.new_line(name = 'ring', components = components)
line.set_particle_ref('proton', p0c = 7e12)
# This small periodic focusing model has a straight survey; 004 uses a closed ring.
line.mechanics.adopt_reference()
mech = env.mechanics
girder = line.mechanics.new_girder('g1', elements = ['q0', 'q1'])
girder.installation.shift_y = INSTALLATION_Y
timings = {'setup_s': perf_counter() - started}

################################################################################
# Use the same named source interface for different inputs
################################################################################
started = perf_counter()
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
if SOURCE == 'harmonic':
    source = xt.mechanics.Harmonic(amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ)
elif SOURCE == 'measured':
    source = xt.mechanics.MeasuredMotion(
        time_s = [0., .002, .004, times[-1]], displacement = [0., AMPLITUDE_M, 0., 0.],
    )
elif SOURCE == 'psd':
    source = xt.mechanics.DisplacementPSD(
        frequency_hz = [0., 1., 2., 40., 41.],
        density = [0., 0., 1e-17, 1e-17, 0.], seed = PSD_SEED,
    )
else:
    raise ValueError('Choose harmonic, measured or psd')
mech.add_source('vertical', source, targets = ['g1'], component = 'y')
for component, amplitude in [('pitch', PITCH_RAD), ('yaw', YAW_RAD), ('roll', ROLL_RAD)]:
    if ROTATION_SOURCE == 'harmonic':
        angular = xt.mechanics.Harmonic(amplitude = amplitude, frequency_hz = FREQUENCY_HZ, unit = 'rad')
    elif ROTATION_SOURCE == 'measured':
        angular = xt.mechanics.MeasuredMotion(time_s = [0., .002, .004, times[-1]],
            displacement = [0., amplitude, 0., 0.], unit = 'rad')
    elif ROTATION_SOURCE == 'psd':
        angular = xt.mechanics.AngularPSD(frequency_hz = [0., 1., 2., 40., 41.],
            density = [0., 0., amplitude**2 / 40., amplitude**2 / 40., 0.],
            seed = PSD_SEED + 1 + ('pitch', 'yaw', 'roll').index(component))
    else:
        raise ValueError('Choose harmonic, measured or psd angular motion')
    mech.add_source(component, angular, targets = ['g1'], component = component)
mech.add_source('slow_drift', xt.mechanics.MeasuredMotion(
    time_s = [times[0], times[-1]], displacement = [0., times[-1] * DRIFT_M_PER_S],
), targets = ['g1'], component = 's')
motion = mech.compute_time_series(time_s = times)
table = motion.get_table('g1')
timings['sample_motion_s'] = perf_counter() - started

################################################################################
# Advance the same physical-time record across split tracking calls
################################################################################
started = perf_counter()
line.build_tracker()
timings['tracker_build_s'] = perf_counter() - started
particles = line.build_particles()
line.mechanics.enable_vibrations(time_series = motion)
started = perf_counter()
line.track(particles, num_turns = N_TURNS // 2)
line.track(particles, num_turns = N_TURNS - N_TURNS // 2)
timings['dynamic_tracking_s'] = perf_counter() - started
line.mechanics.disable_vibrations()
print('Motion inputs and tracking lifecycle\n------------------------------------')
print(f'Source: {SOURCE}; turns: {N_TURNS}; physical time [s]: {particles.time_s:.9g}')
print(f'Installation y after disabling [µm]: {env["q0"].shift_y * 1e6:.6f}')
print(f'Input record payload [KiB]: {sum(v.nbytes for v in motion.values.values()) / 1024:.1f}')
print('Angular inputs compose as rotations. They do not change the installation settings.')
mech.apply(time_s = INSPECT_TIME_S, time_series = motion)
twiss = line.twiss4d()
print(f'Frozen closed orbit at q0 [µm]: {twiss.y[0] * 1e6:.6f}')

################################################################################
# Inspect translations, rotation and slow drift separately
################################################################################
fig, axes = plt.subplots(3, 1, figsize = (8, 7))
axes[0].plot(times, table.shift_y * 1e9, label = SOURCE)
for name, label in [('rot_x_rad', 'Pitch'), ('rot_y_rad', 'Yaw'), ('rot_s_rad', 'Roll')]:
    axes[1].plot(times, table[name] * 1e6, label = label)
axes[2].plot(times, table.shift_s * 1e9)
axes[0].set(xlabel = 'Physical time [s]', ylabel = 'Dynamic y [nm]', xlim = (0, .01),
            title = 'Direct support inputs; installation excluded')
axes[1].set(xlabel = 'Physical time [s]', ylabel = 'Survey angle [µrad]', xlim = (0, .01))
axes[2].set(xlabel = 'Physical time [s]', ylabel = 'Slow s drift [nm]', xlim = (times[0], times[-1]))
axes[0].legend()
axes[1].legend()
fig.tight_layout()
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<24} {seconds:9.4f} s')
plt.show()
