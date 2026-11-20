# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Select a quadrupole through mechanics tables and inspect its composed motion.
# Plot its full history and save a magnified rigid cuboid animation as a GIF.

################################################################################
# Required packages
################################################################################
from pathlib import Path
from time import perf_counter

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
INSTALLATION_Y_M = 20e-6
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

########################################
# Illustrative envelope and display magnification
########################################
CUBOID_SIZE_M      = (.3, .3, MAGNET_LENGTH_M)  # Width x, height y, length s [m]
DISPLACEMENT_SCALE = 2e5
ROTATION_SCALE     = 4e5
N_FRAMES           = 64
GIF_FPS            = 32
SAVE_GIF           = True
GIF_FILE           = Path(__file__).with_name('014_quad_motion.gif')

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
girder = line.mechanics.new_girder('g.shared', elements = ['q1', 'q2'])
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
print('Mechanics tables and one selected quadrupole\n-------------------------------------------')
sources.cols['name model component reference target_names'].show()
girders.cols['name kind parent dynamic_X dynamic_Y dynamic_Z'].show()
quads.cols['name body element_type reference_X static_Y total_Y'].show()

################################################################################
# Follow one selected quadrupole through the full record
################################################################################

########################################
# Access the live Xtrack element through its table row
########################################
selected = quads.rows[QUADRUPOLE]
quad = selected.get_element()
print(f'Selected {QUADRUPOLE}: length {quad.length:g} m; design roll {quad.rot_s_rad:g} rad')

########################################
# Compose the selected element history on the bound record grid
########################################
started = perf_counter()
history = selected.get_time_series()
timings['selected_history_s'] = perf_counter() - started

########################################
# Check the lever arm between the support anchor and magnet entrance
########################################
support_history = motion.get_table(girder.name)
lever_arm_y = history.dynamic_Y - support_history.shift_y
print(f'Element history: {len(history)} samples over {DURATION_S:g} s')
print(f'Largest magnet/support vertical difference [nm]: {np.max(np.abs(lever_arm_y)) * 1e9:.6f}')
print('This is composed element motion, including the support rotation lever arm.')

########################################
# Plot the actual motion with physical units and no display gain
########################################
started = perf_counter()
history_fig, history_axes = history.plot()
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
    print(f'Saved {N_FRAMES} frames: {GIF_FILE}')

########################################
# Verify that inspection never overwrote the deliberate line inputs
########################################
np.testing.assert_equal(quad.shift_x, DELIBERATE_X_M)
np.testing.assert_equal(quad.rot_s_rad, DELIBERATE_ROLL)
np.testing.assert_equal(quad.shift_y, 0.)
print('Tables, histories and animation leave the line inputs untouched.')

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the independently measured operation costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<32} {seconds:9.4f} s')

########################################
# Display the history and the animated envelope
########################################
plt.show()
