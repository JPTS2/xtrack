# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Select a physical magnet through a table row and follow its actual rigid pose.
# Compare its motion with its girder and neighbouring magnet, then animate it.

################################################################################
# Required packages
################################################################################

from pathlib import Path
from time import perf_counter
from tempfile import gettempdir

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Deliberate magnetic reference
########################################
QUADRUPOLE        = 'q1'
MAGNET_LENGTH_M   = .5
DELIBERATE_X_M    = 1e-3
DELIBERATE_ROLL   = .15                 # Existing design roll [rad]

########################################
# Static support installation
########################################
INSTALLATION_Y_M = .2e-6
INSTALLATION_YAW = 2e-6

########################################
# Six prescribed dynamic inputs
########################################
TRANSLATION_PEAK_M = (30e-9, 50e-9, 20e-9)
ANGLE_PEAK_RAD     = (200e-9, 300e-9, 400e-9)
FREQUENCIES_HZ     = (2., 3., 4., 2., 3., 4.)
PHASES_RAD         = (0., .4, .8, .2, .6, 1.)

########################################
# History and inspection sampling
########################################
DURATION_S     = 2.
SAMPLE_STEP_S  = 1 / 128
INSPECT_TIME_S = .125
PRINT_TIMINGS  = False

########################################
# Illustrative envelope and display magnification
########################################
CUBOID_SIZE_M      = (.3, .3, MAGNET_LENGTH_M)  # Width x, height y, length s [m]
DISPLACEMENT_SCALE = 2e5
ROTATION_SCALE     = 4e5
N_FRAMES           = 64
GIF_FPS            = 32
SAVE_GIF           = True
GIF_FILE           = Path(gettempdir()) / 'xtrack_quad_motion.gif'

################################################################################
# Set up the ordinary line and tracker
################################################################################

########################################
# Set up performance monitoring
########################################
timings = {}
started = perf_counter()

########################################
# Construct two distinct quadrupoles on the same line
########################################
env = xt.Environment()
env.new('q1', xt.Quadrupole, length = MAGNET_LENGTH_M, k1 = .2,
        shift_x = DELIBERATE_X_M, rot_s_rad = DELIBERATE_ROLL)
env.new('gap', xt.Drift, length = 1.)
env.new('q2', xt.Quadrupole, length = MAGNET_LENGTH_M, k1 = -.2)
line = env.new_line(name = 'line', components = ['q1', 'gap', 'q2'])
line.set_particle_ref('proton', p0c = 7e12)
timings['line_construction_s'] = perf_counter() - started

########################################
# Prepare the tracker before mechanics performs any survey
########################################
started = perf_counter()
line.build_tracker()
timings['tracker_build_s'] = perf_counter() - started

################################################################################
# Adopt the reference and define the shared support
################################################################################

########################################
# Time reference adoption independently of inspecting its table
########################################
started = perf_counter()
line.mechanics.adopt_reference()
timings['reference_adoption_s'] = perf_counter() - started

########################################
# Access the environment mechanics model
########################################
mech = env.mechanics

########################################
# Attach both magnets to one physical girder
########################################
started = perf_counter()
girder = line.mechanics.new_girder(
    'g.shared', elements = ['q1', 'q2'],
    anchor = xt.Frame.from_survey_angles(Y = -.35, Z = line.get_length() / 2),
)
timings['mechanics_definition_s'] = perf_counter() - started

########################################
# Set installation independently of the dynamic inputs
########################################
girder.installation.shift_y = INSTALLATION_Y_M
girder.installation.rot_y_rad = INSTALLATION_YAW

################################################################################
# Define and sample all six motion coordinates
################################################################################

########################################
# Time the motion-source definitions
########################################
started = perf_counter()

########################################
# Bind separate translations and survey pitch, yaw and roll
########################################
components = ('x', 'y', 's', 'pitch', 'yaw', 'roll')
amplitudes = (*TRANSLATION_PEAK_M, *ANGLE_PEAK_RAD)
for index, (component, amplitude) in enumerate(zip(components, amplitudes)):
    mech.add_source('drive.' + component, xt.mechanics.Harmonic(
        amplitude = amplitude, frequency_hz = FREQUENCIES_HZ[index],
        phase_rad = PHASES_RAD[index], unit = 'm' if index < 3 else 'rad',
    ), targets = ['g.shared'], component = component)
timings['source_definition_s'] = perf_counter() - started

########################################
# Compute the reusable source record on an explicit grid
########################################
started = perf_counter()
times = np.arange(round(DURATION_S / SAMPLE_STEP_S) + 1) * SAMPLE_STEP_S
motion = mech.compute_time_series(time_s = times)
timings['sample_motion_s'] = perf_counter() - started

################################################################################
# Inspect sources, bodies and individual element occurrences
################################################################################

########################################
# Time the optional table inspections separately
########################################
started = perf_counter()

########################################
# Read the source inventory without evaluating new waveforms
########################################
sources = mech.get_source_table()

########################################
# Select girders by their explicit kind
########################################
bodies = mech.get_table(time_s = INSPECT_TIME_S, time_series = motion)
girders = bodies.rows.match(kind = 'girder')

########################################
# Select the two magnets despite their shared mechanical body
########################################
elements = line.mechanics.get_table(time_s = INSPECT_TIME_S, time_series = motion)
quads = elements.rows.match(element_type = 'Quadrupole')
timings['table_inspection_s'] = perf_counter() - started

########################################
# Report the definitions and the selected physical state
########################################
print('Follow one magnet from a table row to its rigid motion\n'
      '------------------------------------------------------')
print('q1 and q2 share one girder whose reference is 0.35 m below the beam.')
print('The existing q1 horizontal offset and design roll are adopted unchanged.')
print(f'Installation y = {INSTALLATION_Y_M * 1e6:g} µm; requested snapshot time = {INSPECT_TIME_S:g} s.')
print(f'Query kind: {quads.query_kind}; live-input state: {quads.live_state}.')
print('\n1. Inspect the six prescribed motion inputs')
sources.cols['name component unit'].show()
print('\n2. Select an element occurrence, even on a shared girder')
quads.cols['name body element_type'].show()
print('Magnet    Installed entrance Y [µm]    Dynamic entrance Y [nm]')
for name in quads.name:
    print(f'{name:<9} {quads["static_Y", name] * 1e6:25.6f}'
          f' {quads["dynamic_Y", name] * 1e9:26.6f}')

################################################################################
# Follow one selected quadrupole through the full record
################################################################################

########################################
# Access the live Xtrack element through its table row
########################################
selected = quads.rows[QUADRUPOLE]
quad = selected.get_element()
print(f'Selected {QUADRUPOLE}: length {quad.length:g} m; design roll {quad.rot_s_rad:g} rad')
print(f'The live element still has shift_y = {quad.shift_y * 1e6:g} µm: querying a pose does not apply it.')

########################################
# Compose the selected element history on the bound record grid
########################################
started = perf_counter()
history = selected.get_time_series()
timings['selected_history_s'] = perf_counter() - started

########################################
# Check the lever arm between the support anchor and magnet entrance
########################################
started = perf_counter()
support_history = girders.rows[girder.name].get_time_series()
timings['support_history_s'] = perf_counter() - started
lever_arm_y = history.dynamic_Y - support_history.dynamic_Y
print(f'\n3. Compose its prescribed history: {len(history)} samples over {DURATION_S:g} s')
print(f'Largest magnet/support vertical difference [nm]: {np.max(np.abs(lever_arm_y)) * 1e9:.6f}')
print('The selected history includes the full rigid lever arm, not just the source displacement.')
print(f'Deliberate x [µm]: {quad.shift_x * 1e6:g}; deliberate roll [mrad]: {quad.rot_s_rad * 1e3:g}.')
print('Dynamic translations and rotations are magnified about the installed pose in the animation.')

########################################
# Plot the actual motion with physical units and no display gain
########################################
started = perf_counter()
neighbour = quads.rows['q2' if QUADRUPOLE == 'q1' else 'q1']
neighbour_history = neighbour.get_time_series()
history_fig, all_axes = plt.subplots(3, 1, figsize = (9, 8),
                                    gridspec_kw = {'height_ratios': [.8, 1.3, 1.3]})
layout_ax = all_axes[0]
history_axes = all_axes[1:]
history_axes[1].sharex(history_axes[0])
magnet_colours = {'q1': 'tab:blue', 'q2': 'tab:orange'}
# The drawing uses installed element frames; the selected element is highlighted.
for name in ('q1', 'q2'):
    matrix = elements['static_matrix', name]
    half_height = CUBOID_SIZE_M[1] / 2
    corners = np.array([[0., -half_height, 0.], [0., -half_height, MAGNET_LENGTH_M],
                        [0., half_height, MAGNET_LENGTH_M], [0., half_height, 0.], [0., -half_height, 0.]])
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    colour = magnet_colours[name]
    layout_ax.fill(world[:, 2], world[:, 1], color = colour,
                   alpha = .3 if name == QUADRUPOLE else .1)
    layout_ax.plot(world[:, 2], world[:, 1], color = colour)
    centre_s = (world[0, 2] + world[1, 2]) / 2
    layout_ax.text(centre_s, .02, name + (' selected' if name == QUADRUPOLE else ''),
                   ha = 'center', fontsize = 9)
    layout_ax.plot([centre_s, centre_s], [-.35, -half_height], color = '.3', lw = 1)
layout_ax.plot([0., line.get_length()], [-.35, -.35], color = '.3', lw = 2)
layout_ax.plot(line.get_length() / 2, -.35, 'kD', ms = 4)
layout_ax.set(xlabel = 'World Z [m]', ylabel = 'World Y [m]',
               title = f'Installed reference layout: {QUADRUPOLE} selected; diamond = girder anchor')
layout_ax.set_ylim(-.45, .25)
for name, values, colour, style in [
    ('Girder reference', support_history.dynamic_Y, '.4', '--'),
    (QUADRUPOLE + ' entrance', history.dynamic_Y, magnet_colours[QUADRUPOLE], '-'),
    (str(neighbour.name[0]) + ' entrance', neighbour_history.dynamic_Y,
     magnet_colours[str(neighbour.name[0])], '-'),
]:
    history_axes[0].plot(times, values * 1e9, color = colour, ls = style, label = name)
history_axes[0].set(ylabel = 'Vertical displacement [nm]')
history_axes[0].set_title('One rigid girder, two distinct magnet-entrance histories', pad = 30)
history_axes[0].margins(y = .30)
history_axes[0].legend(loc = 'lower center', bbox_to_anchor = (.5, 1.02), ncol = 3, fontsize = 8)
for coordinate, label in [('phi', 'Pitch'), ('theta', 'Yaw'), ('psi', 'Roll')]:
    history_axes[1].plot(times, history['dynamic_' + coordinate] * 1e6, label = label)
history_axes[1].set(xlabel = 'Physical time [s]', ylabel = 'Selected magnet rotation [µrad]')
history_axes[1].legend(ncol = 3, loc = 'lower center', bbox_to_anchor = (.5, 1.02), fontsize = 8)
history_axes[0].text(INSPECT_TIME_S, .97, f'  Snapshot at {INSPECT_TIME_S:g} s',
                    transform = history_axes[0].get_xaxis_transform(), va = 'top',
                    fontsize = 8, bbox = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8})
for ax in history_axes:
    ax.axvline(INSPECT_TIME_S, ls = ':', color = '.4')
    ax.grid(alpha = .2)
history_fig.tight_layout()
timings['history_plot_s'] = perf_counter() - started

################################################################################
# Animate a magnified rigid quadrupole envelope
################################################################################

########################################
# Build the animation through the same selected table row
########################################
started = perf_counter()
animation = selected.animate(
    time_s = np.linspace(0., DURATION_S, N_FRAMES, endpoint = False),
    size_m = CUBOID_SIZE_M, baseline = 'static',
    displacement_scale = DISPLACEMENT_SCALE, rotation_scale = ROTATION_SCALE,
    interval_ms = 1000 / GIF_FPS,
)
timings['animation_preparation_s'] = perf_counter() - started

########################################
# Save the GIF separately from pose and tracking costs
########################################
if SAVE_GIF:
    started = perf_counter()
    animation.save(GIF_FILE, writer = 'pillow', fps = GIF_FPS, dpi = 100)
    timings['gif_render_and_encode_s'] = perf_counter() - started
    print(f'\n4. Magnified envelope: saved {N_FRAMES} frames to {GIF_FILE}')

########################################
# Verify that inspection never overwrote the deliberate line inputs
########################################
np.testing.assert_equal(env['q1'].shift_x, DELIBERATE_X_M)
np.testing.assert_equal(env['q1'].rot_s_rad, DELIBERATE_ROLL)
for name in ('q1', 'q2'):
    np.testing.assert_equal(env[name].shift_y, 0.)
print('Tables, histories and animation leave the line inputs untouched.')

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the independently measured operation costs
########################################
if PRINT_TIMINGS:
    print('\nCPU operation costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<32} {seconds:9.4f} s')

########################################
# Display the history and the animated envelope
########################################
plt.show()
