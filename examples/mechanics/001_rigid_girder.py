# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Put two existing quadrupoles on one rigid girder and move them together.
# Compare deliberate, installed and vibrating magnets, then disable and clear.

################################################################################
# Required packages
################################################################################
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Deliberate magnetic geometry
########################################
DELIBERATE_X_M       = 1e-3             # Intentional q1 horizontal offset [m]
DELIBERATE_Y_M       = 3e-6             # Intentional q1 vertical offset [m]
DELIBERATE_ROLL_RAD  = .1               # Intentional q2 roll [rad]

########################################
# Independent installation and vibration
########################################
INSTALLATION_Y_M    = 20e-6             # Static girder displacement [m]
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
# Build and track the deliberate magnetic reference
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
# Track once through the deliberate offsets, before attaching mechanics
########################################
# Every case below uses a fresh copy of this same launch and one traversal.
# These are transport-line exit coordinates, not a closed orbit.
started = perf_counter()
particles = launch.copy()
line.track(particles)
exits['Reference'] = np.array([float(particles.x[0]), float(particles.y[0])])
placements['Reference'] = np.array([env['q1'].shift_y, env['q2'].shift_y])
timings['Reference one-pass tracking'] = perf_counter() - started

################################################################################
# Adopt the existing magnets and apply the girder installation
################################################################################

########################################
# Capture deliberate inputs without retyping them into mechanics
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
# Apply installation before defining any vibration
########################################
started = perf_counter()
mech.apply()
timings['Static pose application'] = perf_counter() - started
placements['Installed'] = np.array([env['q1'].shift_y, env['q2'].shift_y])

########################################
# Track the installed magnets from the same launch
########################################
started = perf_counter()
particles = launch.copy()
line.track(particles)
exits['Installed'] = np.array([float(particles.x[0]), float(particles.y[0])])
timings['Installed one-pass tracking'] = perf_counter() - started

################################################################################
# Apply one common vibration and inspect both magnets
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
# Freeze the magnets at the chosen physical time
########################################
# apply() sets one pose. It does not enable updates inside line.track().
started = perf_counter()
mech.apply(time_s = INSPECT_TIME_S)
timings['Dynamic pose application'] = perf_counter() - started
placements['Vibrating'] = np.array([env['q1'].shift_y, env['q2'].shift_y])

########################################
# Track once through that frozen state
########################################
started = perf_counter()
particles = launch.copy()
line.track(particles)
exits['Vibrating'] = np.array([float(particles.x[0]), float(particles.y[0])])
timings['Vibrating one-pass tracking'] = perf_counter() - started

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
# Disable vibration, then remove mechanics completely
################################################################################

########################################
# Keep installation while removing the dynamic displacement
########################################
started = perf_counter()
line.mechanics.disable_vibrations()
timings['Disable vibrations'] = perf_counter() - started
placements['Disabled'] = np.array([env['q1'].shift_y, env['q2'].shift_y])
particles = launch.copy()
line.track(particles)
exits['Disabled'] = np.array([float(particles.x[0]), float(particles.y[0])])

########################################
# Restore the adopted magnetic inputs and detach mechanics
########################################
started = perf_counter()
mech.clear()
timings['Clear mechanics'] = perf_counter() - started
placements['Cleared'] = np.array([env['q1'].shift_y, env['q2'].shift_y])
particles = launch.copy()
line.track(particles)
exits['Cleared'] = np.array([float(particles.x[0]), float(particles.y[0])])

########################################
# Check rigid inheritance, cleanup and preservation of deliberate inputs
########################################
vibration = placements['Vibrating'] - placements['Installed']
expected = VIBRATION_PEAK_M * np.sin(2 * np.pi * VIBRATION_HZ * INSPECT_TIME_S)
np.testing.assert_allclose(vibration, expected, rtol = 0, atol = 1e-18)
np.testing.assert_allclose(q1_history.dynamic_Y, q2_history.dynamic_Y, rtol = 0, atol = 1e-18)
np.testing.assert_array_equal(placements['Disabled'], placements['Installed'])
np.testing.assert_array_equal(placements['Cleared'], placements['Reference'])
np.testing.assert_array_equal(exits['Disabled'], exits['Installed'])
np.testing.assert_array_equal(exits['Cleared'], exits['Reference'])
np.testing.assert_equal(env['q1'].shift_x, DELIBERATE_X_M)
np.testing.assert_equal(env['q1'].shift_y, DELIBERATE_Y_M)
np.testing.assert_equal(env['q2'].rot_s_rad, DELIBERATE_ROLL_RAD)

################################################################################
# Compare the five placements and their matched beam transport
################################################################################

########################################
# State the reference and the two independent mechanical inputs
########################################
print('Two quadrupoles on one rigid girder\n'
      '----------------------------------')
print('Line: q1 (0.2 m), drift (2 m), q2 (0.2 m).')
print(f'Adopted q1 x / y [µm]: {reference["shift_x", "q1"] * 1e6:.3f}'
      f' / {reference["shift_y", "q1"] * 1e6:.3f}')
print(f'q2 design roll, preserved after clear [mrad]: {env["q2"].rot_s_rad * 1e3:.3f}')
print(f'Static girder y [µm]: {INSTALLATION_Y_M * 1e6:.3f}')
print(f'Common vibration: {VIBRATION_PEAK_M * 1e9:g} nm peak at {VIBRATION_HZ:g} Hz.')
print(f'Frozen inspection time [ms]: {INSPECT_TIME_S * 1e3:g}')

########################################
# Show what each operation leaves in the actual element inputs
########################################
print('\nVertical magnet alignment\n-------------------------')
print(f'{"State":<12} {"q1 y [µm]":>12} {"q2 y [µm]":>12}')
for label, values in placements.items():
    print(f'{label:<12} {values[0] * 1e6:12.3f} {values[1] * 1e6:12.3f}')
print(f'Vibrating minus installed [nm]: q1 {vibration[0] * 1e9:+.3f};'
      f' q2 {vibration[1] * 1e9:+.3f}.')
print('Disable removes vibration and keeps installation.')
print('Clear removes installation too; the adopted q1 offsets and q2 roll survive.')

########################################
# Isolate the vibration response with the same launch in every case
########################################
print('\nMatched one-pass tracking\n-------------------------')
print('Each case starts at x = px = y = py = 0 with a 7 TeV/c proton.')
print(f'{"State":<12} {"Exit x [µm]":>14} {"Exit y [µm]":>14}')
for label, values in exits.items():
    print(f'{label:<12} {values[0] * 1e6:+14.6f} {values[1] * 1e6:+14.6f}')
beam_change = exits['Vibrating'] - exits['Installed']
print(f'Vibration-only change in exit y [nm]: {beam_change[1] * 1e9:+.6f}')
print('Disabled reproduces installed tracking; cleared reproduces reference tracking.')

################################################################################
# Plot installation separately from the much smaller common vibration
################################################################################

########################################
# Compare actual element inputs through the five lifecycle states
########################################
fig, axes = plt.subplots(1, 2, figsize = (11, 4.6), gridspec_kw = {'width_ratios': [1.25, 1]})
labels = list(placements)
y = np.arange(len(labels))
for index, (name, color, offset) in enumerate([
    ('q1', 'tab:blue', -.17), ('q2', 'tab:orange', .17),
]):
    values = np.array([placements[label][index] for label in labels]) * 1e6
    bars = axes[0].barh(y + offset, values, height = .30, color = color, label = name)
    axes[0].bar_label(bars, fmt = '%.3f', padding = 4, fontsize = 9)
axes[0].set(
    yticks = y, yticklabels = labels, xlabel = 'Vertical magnet alignment [µm]',
    title = 'What remains after each operation?',
)
axes[0].invert_yaxis()
axes[0].margins(x = .28)
axes[0].legend(loc = 'lower right')

########################################
# Resolve the nanometre vibration about each installed magnet
########################################
axes[1].plot(times * 1e3, q1_history.dynamic_Y * 1e9,
             color = 'tab:blue', label = 'q1')
axes[1].plot(times[::8] * 1e3, q2_history.dynamic_Y[::8] * 1e9,
             'o', ms = 5, mfc = 'none', color = 'tab:orange', label = 'q2')
axes[1].axvline(INSPECT_TIME_S * 1e3, ls = ':', color = '0.4', label = 'Applied time')
axes[1].set(
    xlabel = 'Physical time [ms]', ylabel = 'Displacement from installed pose [nm]',
    title = 'Both magnets inherit the same vibration',
)
axes[1].margins(y = .25)
axes[1].legend(loc = 'upper center', ncol = 3, fontsize = 9)
for ax in axes:
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
