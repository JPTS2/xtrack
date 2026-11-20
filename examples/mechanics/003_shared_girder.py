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
FORWARD_RETURN_M  = 100000.
BACKWARD_RETURN_M = 180000.

########################################
# Common world displacement
########################################
AMPLITUDE_M  = 50e-9
FREQUENCY_HZ = 500.                   # Makes the distinct passage sampling visible

########################################
# Shared physical origin and independent beam clocks
########################################
TIME_ORIGIN_S  = 0.                   # Shared physical origin, not a shared turn counter
N_TURNS        = 32
SAMPLE_STEP_S  = 1e-5
N_SAMPLES      = 8192
COMPARE_TIME_S = .002
PRINT_TIMINGS  = False

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
# Verify that the two local descriptions put the magnet centre at one world point
########################################
world_centres = []
for beam_line, name in [(forward, 'q.forward'), (backward, 'q.backward')]:
    frame = beam_line.mechanics.get_element_poses(time_s = COMPARE_TIME_S, time_series = motion)[name]
    world_centres.append(frame.XYZ + frame.matrix[:3, :3] @ [0., 0., .5])
np.testing.assert_allclose(world_centres[0], world_centres[1], atol = 1e-14, rtol = 0)
shared_x_nm = world_centres[0][0] * 1e9

########################################
# Describe one physical displacement in the two beam coordinate systems
########################################
print('One girder, two beam coordinate systems\n--------------------------------------')
print('The two zero-strength test elements occupy the same world segment.')
print('Forward local s follows +Z; backward local s follows -Z.')
print('Backward local x therefore points towards -X. Both local y axes point towards +Y.')
print(f'At the same physical time t = {COMPARE_TIME_S * 1e3:g} ms:')
print('Beam        World centre X [nm]    Line-local shift x [nm]')
for label, name in [('Forward', 'q.forward'), ('Backward', 'q.backward')]:
    print(f'{label:<11} {shared_x_nm:+20.3f} {env[name].shift_x * 1e9:+26.3f}')
print('Opposite local signs describe the same moved magnet, not opposite physical motion.')
print('\nIndependent passage sampling\n----------------------------')
print('Beam       Turn spacing [ms]    Last sampled physical time [ms]')
for label, (sampled_times, _) in histories.items():
    print(f'{label:<10} {np.diff(sampled_times).mean() * 1e3:16.6f} {sampled_times[-1] * 1e3:33.6f}')
print('Each beam reads one immutable physical field at its own passage times.')
print('This example isolates geometry and clocks; causal shared responses/controllers require scheduling.')

################################################################################
# Draw the shared geometry before comparing its coordinate descriptions
################################################################################

########################################
# Show the common support and the reversed local axes
########################################
fig, axes = plt.subplots(2, 2, figsize = (10, 7))
ax = axes[0, 0]
nominal_box = np.array([[0., -.08], [1., -.08], [1., .08], [0., .08], [0., -.08]])
moved_box = nominal_box + [0., shared_x_nm * 1e-3]  # ×1,000,000 geometry gain
ax.plot(*nominal_box.T, '--', color = '0.65', label = 'Installed shared support')
ax.fill(*moved_box.T, color = 'tab:blue', alpha = .15)
ax.plot(*moved_box.T, color = 'tab:blue', label = 'Same support displaced towards +X')
for z0, z1, height, colour, text in [(0., 1., -.16, 'tab:blue', 'Forward: local +s'),
                                    (1., 0., -.24, 'tab:orange', 'Backward: local +s')]:
    ax.annotate('', xy = (z1, height), xytext = (z0, height),
                arrowprops = {'arrowstyle': '->', 'color': colour, 'lw': 1.5})
    ax.text(.5, height - .025, text, color = colour, ha = 'center', va = 'top', fontsize = 9)
ax.annotate('Forward +x', xy = (.35, .23), xytext = (.35, .02), ha = 'center', fontsize = 9,
            arrowprops = {'arrowstyle': '->', 'color': 'tab:blue'})
ax.annotate('Backward +x', xy = (.72, -.07), xytext = (.72, .2), ha = 'center', fontsize = 9,
            arrowprops = {'arrowstyle': '->', 'color': 'tab:orange'})
ax.set(title = 'One physical support; opposite beam axes', xlabel = 'World Z [m]',
       ylabel = 'World X [m], motion ×1,000,000', xlim = (-.1, 1.1), ylim = (-.33, .30))
# Keep the shared-body key outside the opposed-axis schematic.
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc = 'lower center', ncol = 2, fontsize = 8)

########################################
# Put both sampled histories into world coordinates on equal footing
########################################
curve_time = np.linspace(0, max(t[-1] for t, _ in histories.values()), 1200)
curve_x = source.evaluate(curve_time, np.array([girder.nominal_frame.XYZ]))[:, 0, 0]
axes[0, 1].plot(curve_time * 1e3, curve_x * 1e9, color = '0.7', lw = 1, label = 'Prescribed world motion')
for label, sign, marker in [('Forward', 1., 'o'), ('Backward', -1., 's')]:
    sampled_times, local_x = histories[label]
    axes[0, 1].plot(sampled_times * 1e3, sign * local_x * 1e9, marker,
                    ms = 4, mfc = 'none', label = label + ' passages')
    axes[1, 0].plot(sampled_times * 1e3, local_x * 1e9, '.-', label = label)
axes[0, 1].set(title = 'Both beams measure the same world displacement', xlabel = 'Physical time [ms]', ylabel = 'World X displacement [nm]')
axes[1, 0].set(title = 'The line-local horizontal coordinates have opposite signs', xlabel = 'Physical time [ms]', ylabel = 'Local horizontal alignment [nm]')
for ax in (axes[0, 1], axes[1, 0]):
    ax.legend(fontsize = 8)
    ax.grid(alpha = .2)

########################################
# Compare both descriptions at exactly the same time
########################################
labels = ['Forward', 'Backward']
local_values = [env['q.forward'].shift_x * 1e9, env['q.backward'].shift_x * 1e9]
xx = np.arange(2)
axes[1, 1].bar(xx - .18, [shared_x_nm] * 2, width = .34, label = 'World X displacement')
axes[1, 1].bar(xx + .18, local_values, width = .34, label = 'Local x alignment')
axes[1, 1].axhline(0., color = '0.5', lw = .7)
axes[1, 1].set(xticks = xx, xticklabels = labels, ylabel = 'Displacement [nm]',
                title = f'Same physical time: t = {COMPARE_TIME_S * 1e3:g} ms')
axes[1, 1].legend(fontsize = 8)
fig.tight_layout(rect = (0., .045, 1., 1.), h_pad = 2.)

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
if PRINT_TIMINGS:
    print('\nCPU operation costs\n-------------------')
if PRINT_TIMINGS:
    for name, seconds in timings.items():
        print(f'{name:<30} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
