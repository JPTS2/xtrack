# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Sample one prescribed physical girder from two beams with different turn times.
# Separate the shared world motion from opposite local coordinates and clocks.

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
# Two line lengths
########################################
FORWARD_RETURN_M  = 100.
BACKWARD_RETURN_M = 180.

########################################
# Common world displacement
########################################
AMPLITUDE_M  = 50e-9
FREQUENCY_HZ = 1e5                    # High on purpose: turn numbers differ visibly

########################################
# Shared physical origin and independent beam clocks
########################################
TIME_ORIGIN_S  = 0.                   # Shared physical origin, not a shared turn counter
N_TURNS        = 64
SAMPLE_STEP_S  = 1e-8
N_SAMPLES      = 8192
COMPARE_TIME_S = 1e-5

################################################################################
# Set up two ordinary line occurrences
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

########################################
# Time line construction
########################################
started = perf_counter()

########################################
# Construct distinct elements and line sequences
########################################
env = xt.Environment()
env.new('q.forward', xt.Quadrupole, length = 1., k1 = 0.)
env.new('q.backward', xt.Quadrupole, length = 1., k1 = 0.)
env.new('return.forward', xt.Drift, length = FORWARD_RETURN_M)
env.new('return.backward', xt.Drift, length = BACKWARD_RETURN_M)
forward = env.new_line(name = 'forward', components = ['q.forward', 'return.forward'])
backward = env.new_line(name = 'backward', components = ['q.backward', 'return.backward'])

########################################
# Give each line its reference particle
########################################
for line in (forward, backward):
    line.set_particle_ref('proton', p0c = 7e12)

timings['line_construction_s'] = perf_counter() - started

########################################
# Prepare and time the tracker before mechanics queries
########################################
for label, beam_line in [('forward', forward), ('backward', backward)]:
    started = perf_counter()
    beam_line.build_tracker()
    timings[label + '_tracker_build_s'] = perf_counter() - started

################################################################################
# Place both line surveys in one physical world
################################################################################

########################################
# Time mechanics definition
########################################
started = perf_counter()

########################################
# Set the backward survey origin and direction
########################################
backward.mechanics.set_reference_frame(xt.Frame.from_survey_angles(Z = 1., theta = np.pi))

########################################
# Access the shared mechanics model
########################################
mech = env.mechanics

########################################
# Create the shared physical girder on the forward line
########################################
girder = forward.mechanics.new_girder('shared', elements = ['q.forward'])

########################################
# Attach the backward occurrence to the same girder
########################################
backward.mechanics.attach(girder, elements = ['q.backward'])

timings['mechanics_definition_s'] = perf_counter() - started

################################################################################
# Define and sample the common motion
################################################################################

########################################
# Time source definition
########################################
started = perf_counter()

########################################
# Specify a world-polarized traveling wave
########################################
source = xt.mechanics.PlaneWave(
    amplitude = AMPLITUDE_M, frequency_hz = FREQUENCY_HZ, speed_m_s = 2e4,
    direction = (0., 0., 1.), polarization = (1., 0., 0.), phase_rad = np.pi / 2,
)

########################################
# Bind the wave to the physical girder
########################################
mech.add_source('surface_wave', source, targets = ['shared'])

timings['source_definition_s'] = perf_counter() - started

########################################
# Time sample motion
########################################
started = perf_counter()

########################################
# Sample one record for both beams
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
motion = mech.compute_time_series(time_s = times)

timings['sample_motion_s'] = perf_counter() - started

################################################################################
# Track each beam with its own physical clock
################################################################################

########################################
# Prepare independent measurement histories
########################################
histories = {}

########################################
# Repeat the same procedure for each line
########################################
for label, line, element in [('Forward', forward, 'q.forward'), ('Backward', backward, 'q.backward')]:

    ########################################
    # Create the launch and recording lists
    ########################################
    particles = line.build_particles()
    sampled_times, sampled_x = [], []

    ########################################
    # Enable replay with the explicit common time origin
    ########################################
    line.mechanics.enable_vibrations(time_series = motion, time_origin_s = TIME_ORIGIN_S)

    ########################################
    # Time the beam traversal
    ########################################
    started = perf_counter()

    ########################################
    # Track and record this line's sampled times and offsets
    ########################################
    line.track(particles, num_turns = N_TURNS, log = xt.Log(
        time_s = lambda ll, pp: sampled_times.append(pp.time_s) or pp.time_s,
        shift_x = lambda ll, pp: sampled_x.append(float(ll[element].shift_x)) or float(ll[element].shift_x),
    ))
    timings[label.lower() + '_tracking_s'] = perf_counter() - started

    ########################################
    # Disable this line and retain its history
    ########################################
    line.mechanics.disable_vibrations()
    histories[label] = (np.array(sampled_times) + TIME_ORIGIN_S, np.array(sampled_x))

################################################################################
# Compare the same physical instant explicitly
################################################################################

########################################
# Apply the selected time to both line-local views
########################################
forward.mechanics.apply(time_s = COMPARE_TIME_S, time_series = motion)
backward.mechanics.apply(time_s = COMPARE_TIME_S, time_series = motion)

########################################
# Check the sign change under opposite traversal
########################################
np.testing.assert_allclose(env['q.forward'].shift_x, -env['q.backward'].shift_x, atol = 1e-15, rtol = 0)

########################################
# Report the different turn intervals and local coordinates
########################################
print('Shared body and separate beam clocks\n------------------------------------')
print('Beam       Turn spacing [µs]    Last sampled time [µs]')
for label, (sampled_times, _) in histories.items():
    print(f'{label:<10} {np.diff(sampled_times).mean() * 1e6:16.6f} {sampled_times[-1] * 1e6:25.6f}')
print(f'Same-time local x forward / backward [nm]: {env["q.forward"].shift_x * 1e9:+.6f} / '
      f'{env["q.backward"].shift_x * 1e9:+.6f}')
print('Both lines query one immutable field; running one does not advance the other.')
print('This does not schedule causal shared filters, beam encounters or controllers.')

################################################################################
# Compare physical time with turn number
################################################################################

########################################
# Prepare world-frame and line-local panels
########################################
fig, axes = plt.subplots(2, 1, figsize = (8, 6.5))

########################################
# Evaluate the continuous shared world motion
########################################
curve_time = np.linspace(0, max(t[-1] for t, _ in histories.values()), 1000)
curve_x = source.evaluate(curve_time, np.array([girder.nominal_frame.XYZ]))[:, 0, 0]
axes[0].plot(curve_time * 1e6, curve_x * 1e9, color = '0.6', label = 'Shared world motion')

########################################
# Overlay both beam histories with their coordinate signs
########################################
for label, sign in [('Forward', 1.), ('Backward', -1.)]:
    sampled_times, local_x = histories[label]
    axes[0].plot(sampled_times * 1e6, sign * local_x * 1e9, '.', label = label)
    axes[1].plot(np.arange(N_TURNS), local_x * 1e9, '.-', label = label)

########################################
# Label physical time and independent turn indices
########################################
axes[0].set(xlabel = 'Physical time [µs]', ylabel = 'World X [nm]')
axes[1].set(xlabel = 'Turn index in each beam', ylabel = 'Line-local x [nm]')
axes[0].legend()
axes[1].legend()
fig.tight_layout()

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<30} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
