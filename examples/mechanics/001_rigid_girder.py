# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Put two existing quadrupoles on one rigid girder and move them together.
# Add a static installation error and vibration to the design placement.
# Remove vibration, then clear mechanics and recover the original design.

################################################################################
# Required packages
################################################################################
from textwrap import fill
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Design placement: deliberate magnet offsets and roll
########################################
DELIBERATE_X_M       = 1e-3             # Intentional q1 horizontal offset [m]
DELIBERATE_Y_M       = .1e-6            # Intentional q1 vertical offset [m]
DELIBERATE_ROLL_RAD  = .1               # Intentional q2 roll [rad]

########################################
# Static installation error and dynamic vibration
########################################
INSTALLATION_Y_M    = .2e-6             # Static girder installation error [m]
VIBRATION_PEAK_M    = 50e-9             # Common vertical sine amplitude [m]
VIBRATION_HZ        = 5.
INSPECT_TIME_S      = .05               # A vibration maximum with these inputs

########################################
# Read-only history and execution context
########################################
HISTORY_DURATION_S  = .4                # Two vibration periods with these inputs
N_HISTORY_SAMPLES   = 161
CONTEXT             = xo.ContextCpu()

################################################################################
# Build and track the design placement
################################################################################

########################################
# Set up performance monitoring and comparison results
########################################
timings     = {}
placements  = {}
exits       = {}

########################################
# Construct two thick quadrupoles and a separating drift
########################################
started = perf_counter()
env = xt.Environment()
env.new('q1', xt.Quadrupole, length = .2, k1 = .2,
        shift_x = DELIBERATE_X_M, shift_y = DELIBERATE_Y_M)
env.new('gap', xt.Drift, length = 2.)
env.new('q2', xt.Quadrupole, length = .2, k1 = -.2,
        rot_s_rad = DELIBERATE_ROLL_RAD)
line = env.new_line(name = 'line', components = ['q1', 'gap', 'q2'])
line.set_particle_ref('proton', p0c = 7e12)
timings['Line construction'] = perf_counter() - started

########################################
# Prepare the tracker and the common on-axis launch
########################################
started = perf_counter()
line.build_tracker(_context = CONTEXT)
launch = line.build_particles()
timings['Tracker and launch preparation'] = perf_counter() - started

########################################
# Track the design placement before adding any mechanical errors
########################################
# Every case below uses a fresh copy of this same launch and one traversal.
# These are transport-line exit coordinates, not a closed orbit.
started = perf_counter()
particles = launch.copy()
line.track(particles)
exits['Design'] = np.array([float(particles.x[0]), float(particles.y[0])])
placements['Design'] = np.array([env['q1'].shift_y, env['q2'].shift_y])
timings['Design one-pass tracking'] = perf_counter() - started

################################################################################
# Adopt the design placement and add a static installation error
################################################################################

########################################
# Adopt the existing design offsets and roll without retyping them
########################################
started = perf_counter()
mech = env.mechanics
line.mechanics.adopt_reference()
reference = line.mechanics.get_reference_table()
timings['Reference adoption and inspection'] = perf_counter() - started

########################################
# Attach both quadrupoles to one rigid girder
########################################
started = perf_counter()
girder = line.mechanics.new_girder('girder', elements = ['q1', 'q2'])
girder.installation.shift_y = INSTALLATION_Y_M
timings['Girder and installation definition'] = perf_counter() - started

########################################
# Apply the static installation error to the design placement
########################################
started = perf_counter()
mech.apply()
timings['Static pose application'] = perf_counter() - started
placements['Static installation error'] = np.array([env['q1'].shift_y, env['q2'].shift_y])

########################################
# Track with design offsets and static installation error
########################################
started = perf_counter()
particles = launch.copy()
line.track(particles)
exits['Static installation error'] = np.array([float(particles.x[0]), float(particles.y[0])])
timings['Static installation error: tracking'] = perf_counter() - started

################################################################################
# Add vibration to the design placement and static installation error
################################################################################

########################################
# Bind a vertical sine displacement to the girder
########################################
# The girder moves both magnets. q1 keeps its deliberate offset and q2 its roll.
# Rotation and relative magnet motion are introduced in later lessons.
started = perf_counter()
mech.add_source('vertical', xt.mechanics.Harmonic(
    amplitude = VIBRATION_PEAK_M, frequency_hz = VIBRATION_HZ,
), targets = ['girder'], component = 'y')
timings['Vibration source definition'] = perf_counter() - started

########################################
# Apply the vibration displacement at the selected time
########################################
# apply() sets one pose. It does not enable updates inside line.track().
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S)
timings['Dynamic pose application'] = perf_counter() - started
placements['Vibration applied'] = np.array([env['q1'].shift_y, env['q2'].shift_y])

########################################
# Track once through that frozen state
########################################
started = perf_counter()
particles = launch.copy()
line.track(particles)
exits['Vibration applied'] = np.array([float(particles.x[0]), float(particles.y[0])])
timings['Vibration pose: one-pass tracking'] = perf_counter() - started

########################################
# Read each magnet's history without changing the applied pose
########################################
# dynamic_Y is displacement from the installed pose, in world metres.
# In this straight line, world Y and the vertical alignment direction coincide.
started = perf_counter()
times = np.linspace(0., HISTORY_DURATION_S, N_HISTORY_SAMPLES)
elements = line.mechanics.get_table(time_s = INSPECT_TIME_S)
q1_history = elements.rows['q1'].get_time_series(time_s = times)
q2_history = elements.rows['q2'].get_time_series(time_s = times)
timings['Read-only element histories'] = perf_counter() - started

################################################################################
# Remove vibration, then clear mechanics and restore the design placement
################################################################################

########################################
# Remove vibration while retaining the static installation error
########################################
started = perf_counter()
line.mechanics.disable_vibrations()
timings['Disable vibrations'] = perf_counter() - started
placements['Vibration removed'] = np.array([env['q1'].shift_y, env['q2'].shift_y])
particles = launch.copy()
line.track(particles)
exits['Vibration removed'] = np.array([float(particles.x[0]), float(particles.y[0])])

########################################
# Clear mechanics and restore the original design offsets and roll
########################################
started = perf_counter()
mech.clear()
timings['Clear mechanics'] = perf_counter() - started
placements['Mechanics cleared'] = np.array([env['q1'].shift_y, env['q2'].shift_y])
particles = launch.copy()
line.track(particles)
exits['Mechanics cleared'] = np.array([float(particles.x[0]), float(particles.y[0])])

########################################
# Check rigid inheritance, cleanup and preservation of deliberate inputs
########################################
vibration = placements['Vibration applied'] - placements['Static installation error']
expected = VIBRATION_PEAK_M * np.sin(2 * np.pi * VIBRATION_HZ * INSPECT_TIME_S)
np.testing.assert_allclose(vibration, expected, rtol = 0, atol = 1e-18)
np.testing.assert_allclose(q1_history.dynamic_Y, q2_history.dynamic_Y, rtol = 0, atol = 1e-18)
np.testing.assert_array_equal(placements['Vibration removed'], placements['Static installation error'])
np.testing.assert_array_equal(placements['Mechanics cleared'], placements['Design'])
np.testing.assert_array_equal(exits['Vibration removed'], exits['Static installation error'])
np.testing.assert_array_equal(exits['Mechanics cleared'], exits['Design'])
np.testing.assert_equal(env['q1'].shift_x, DELIBERATE_X_M)
np.testing.assert_equal(env['q1'].shift_y, DELIBERATE_Y_M)
np.testing.assert_equal(env['q2'].rot_s_rad, DELIBERATE_ROLL_RAD)

################################################################################
# Compare design, installation error, vibration and the two cleanup operations
################################################################################

########################################
# Describe the design placement, static error and vibration
########################################
print('Two quadrupoles on one rigid girder\n'
      '----------------------------------')
print('Line: q1 (0.2 m), drift (2 m), q2 (0.2 m).')
print(f'Design q1 offsets x / y [µm]: {reference["shift_x", "q1"] * 1e6:.3f}'
      f' / {reference["shift_y", "q1"] * 1e6:.3f}')
print(f'Design q2 roll, preserved after mechanics is cleared [mrad]: {env["q2"].rot_s_rad * 1e3:.3f}')
print(f'Static girder installation error y [µm]: {INSTALLATION_Y_M * 1e6:.3f}')
print(f'Common vibration: {VIBRATION_PEAK_M * 1e9:g} nm peak at {VIBRATION_HZ:g} Hz.')
print(f'Vibration displacement applied at t = {INSPECT_TIME_S:g} s.')
print('Each traversal holds that placement fixed; automatic updates are introduced in 002.')

########################################
# Show what each operation leaves in the actual element inputs
########################################
state_labels = {
    'Design': 'Design placement (deliberate offsets and roll)',
    'Static installation error': 'Design + static installation error',
    'Vibration applied': f'Design + installation error + vibration (t = {INSPECT_TIME_S:g} s)',
    'Vibration removed': 'Vibration removed; static installation error retained',
    'Mechanics cleared': 'Mechanics cleared; design placement restored',
}
state_width = max(len(label) for label in state_labels.values())
print('\nVertical magnet placement through the mechanics lifecycle\n'
      '--------------------------------------------------------')
print(f'{"Magnet placement state":<{state_width}} {"q1 y [µm]":>12} {"q2 y [µm]":>12}')
for label, values in placements.items():
    print(f'{state_labels[label]:<{state_width}} {values[0] * 1e6:12.3f} {values[1] * 1e6:12.3f}')
print(f'Vibration displacement added to the installed placement [nm]: q1 {vibration[0] * 1e9:+.3f};'
      f' q2 {vibration[1] * 1e9:+.3f}.')
print('Removing vibration leaves the design offsets and static installation error in place.')
print('Clearing mechanics also removes the installation error and restores the design placement.')

########################################
# Isolate the vibration response with the same launch in every case
########################################
print('\nMatched one-pass tracking\n-------------------------')
print('Each case starts at x = px = y = py = 0 with a 7 TeV/c proton.')
print(f'{"Magnet placement state":<{state_width}} {"Exit x [µm]":>14} {"Exit y [µm]":>14}')
for label, values in exits.items():
    print(f'{state_labels[label]:<{state_width}} {values[0] * 1e6:+14.6f} {values[1] * 1e6:+14.6f}')
beam_change = exits['Vibration applied'] - exits['Static installation error']
print(f'Vibration-only change in exit y [nm]: {beam_change[1] * 1e9:+.6f}')
print('After vibration is removed, tracking matches the static installation-error case.')
print('After mechanics is cleared, tracking matches the original design placement.')

################################################################################
# Plot installation separately from the much smaller common vibration
################################################################################

########################################
# Compare actual element inputs through the five lifecycle states
########################################
fig = plt.figure(figsize = (12.5, 5.6))
grid = fig.add_gridspec(2, 2, width_ratios = [1.1, 1])
placement_ax = fig.add_subplot(grid[:, 0])
q1_ax = fig.add_subplot(grid[0, 1])
q2_ax = fig.add_subplot(grid[1, 1], sharex = q1_ax, sharey = q1_ax)
labels = list(placements)
y = np.arange(len(labels))
for index, (name, color, offset) in enumerate([
    ('q1', 'tab:blue', -.17), ('q2', 'tab:orange', .17),
]):
    values = np.array([placements[label][index] for label in labels]) * 1e6
    bars = placement_ax.barh(y + offset, values, height = .30, color = color, label = name)
    placement_ax.bar_label(bars, fmt = '%.3f', padding = 4, fontsize = 9)
placement_ax.set(
    yticks = y, yticklabels = [fill(state_labels[label], width = 32) for label in labels],
    xlabel = 'Vertical magnet placement [µm]',
    title = 'Design, installation error and vibration',
)
placement_ax.tick_params(axis = 'y', labelsize = 9)
placement_ax.invert_yaxis()
placement_ax.margins(x = .28)
placement_ax.legend(loc = 'lower right')

########################################
# Resolve the nanometre vibration about each installed magnet
########################################
# The histories coincide. Matching panels expose both complete curves with
# the same sampling, line style and axis limits, without one hiding the other.
for ax, name, history, color in [
    (q1_ax, 'q1', q1_history, 'tab:blue'),
    (q2_ax, 'q2', q2_history, 'tab:orange'),
]:
    ax.plot(times * 1e3, history.dynamic_Y * 1e9, color = color)
    ax.axvline(INSPECT_TIME_S * 1e3, ls = ':', color = '0.4',
               label = f't = {INSPECT_TIME_S:g} s')
    ax.set(
        ylabel = 'Vibration displacement [nm]',
        title = f'{name}: vibration about the installed placement',
    )
    ax.title.set_fontsize(11)
    ax.margins(y = .30)
    ax.legend(loc = 'upper right', fontsize = 9)
q1_ax.tick_params(labelbottom = False)
q2_ax.set_xlabel('Physical time [ms]')
placement_ax.title.set_fontsize(11)
for ax in (placement_ax, q1_ax, q2_ax):
    ax.grid(axis = 'x', alpha = .2)
    ax.set_axisbelow(True)
fig.tight_layout()

################################################################################
# Review the separate setup, application and inspection costs
################################################################################

########################################
# Report physical operations in milliseconds
########################################
print('\nCPU operation costs\n-------------------')
print(f'{"Operation":<38} {"Time [ms]":>10}')
for label, seconds in timings.items():
    print(f'{label:<38} {seconds * 1e3:10.3f}')
print('One-pass costs include a fresh launch copy and first-use dispatch where applicable.')
print('Histories are read-only diagnostics; these timings are not a throughput comparison.')
print('No automatic vibration updates run in this lesson. Example 002 introduces them.')

########################################
# Display the two physical comparisons
########################################
plt.show()
