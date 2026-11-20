# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Does a cross-coupled support response move the mounted magnet the same way?
# Separate mechanical cross coupling from geometric transport to a mounted magnet.

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
# Line and magnet selection
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLE   = 'arc.s1.c1.qf.in'

########################################
# Six-coordinate input amplitudes and frequencies
########################################
TRANSLATION_PEAK_M = (10e-9, 50e-9, 5e-9)
ANGLE_PEAK_RAD     = (2e-9, 3e-9, 4e-9)         # Survey pitch, yaw, roll
DRIVE_HZ           = (2., 5., 3., 5., 7., 11.)

########################################
# Response poles and mixed-unit couplings
########################################
RESONANCE_HZ         = (12., 10., 25., 15., 18., 22.)
DAMPING              = .15
YAW_TO_X_M_PER_RAD   = 1.5                             # Supplied response coupling, same port
Y_TO_PITCH_RAD_PER_M = .2

########################################
# Independent geometric lever arm
########################################
MAGNET_MOUNT_M = (.1, .5, 1.)         # Lever arm, handled separately by transport

########################################
# Record sampling
########################################
N_SAMPLES     = 2048
SAMPLE_STEP_S = 1 / 256

########################################
# Frozen inspection and optional animation
########################################
INSPECT_TIME_S = .125
MAKE_ANIMATION = False

########################################
# Optional performance diagnostics
########################################
PRINT_TIMINGS = False

################################################################################
# Set up the ordinary Xtrack line
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

started = perf_counter()

########################################
# Locate and load the demonstration ring
########################################
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line

timings['line_load_and_copy_s'] = perf_counter() - started

########################################
# Prepare and time the tracker before mechanics queries
########################################
started = perf_counter()
line.build_tracker()
timings['tracker_build_s'] = perf_counter() - started

################################################################################
# Define the support and its magnet mount
################################################################################

########################################
# Access the shared mechanics model
########################################
mech = env.mechanics

started = perf_counter()

########################################
# Adopt existing offsets as the deliberate reference
########################################
line.mechanics.adopt_reference()

timings['reference_adoption_s'] = perf_counter() - started

started = perf_counter()

########################################
# Survey the magnet's nominal reference port
########################################
nominal = line.survey(include_element_frames = True)

timings['geometry_selection_s'] = perf_counter() - started
started = perf_counter()

magnet_frame = nominal.get_frame(QUADRUPOLE, which = 'elem_start')

########################################
# Specify the parent-local mount and lever arm
########################################
mount = xt.Frame.from_survey_angles(X = MAGNET_MOUNT_M[0], Y = MAGNET_MOUNT_M[1], Z = MAGNET_MOUNT_M[2])

########################################
# Place the support and attach the existing magnet
########################################
girder = mech.new_girder('support', anchor = magnet_frame @ mount.inverse())
magnet = mech.new_body('magnet', kind = 'magnet', parent = girder, mount = mount, elements = [QUADRUPOLE], line = line)

timings['mechanics_definition_s'] = perf_counter() - started

################################################################################
# Prepare a six-coordinate prescribed drive
################################################################################

started = perf_counter()

########################################
# Construct the physical time grid and coordinate amplitudes
########################################
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
amplitude = np.array([*TRANSLATION_PEAK_M, *ANGLE_PEAK_RAD])
drive_values = amplitude[None, None, :] * np.sin(2 * np.pi * times[:, None, None] * np.array(DRIVE_HZ))

########################################
# Create the named input motion record
########################################
drive = xt.mechanics.RigidMotion(times, drive_values, ('support',))

################################################################################
# Specify the complex response matrix
################################################################################

########################################
# Prepare frequency bins and the six-by-six table
########################################
frequency = np.fft.rfftfreq(N_SAMPLES, SAMPLE_STEP_S)
matrix = np.zeros((len(frequency), 6, 6), dtype = complex)

########################################
# Set the independent coordinate responses
########################################
for index, resonance in enumerate(RESONANCE_HZ):
    matrix[:, index, index] = xt.mechanics.BaseExcitedOscillator(resonance, DAMPING).evaluate(frequency)
diagonal_matrix = matrix.copy()

########################################
# Add yaw-to-x and y-to-pitch response couplings in SI units
########################################
matrix[:, 0, 4] = YAW_TO_X_M_PER_RAD * matrix[:, 0, 0]
matrix[:, 3, 1] = Y_TO_PITCH_RAD_PER_M * matrix[:, 3, 3]

########################################
# Create the tabulated response at the declared port
########################################
response = xt.mechanics.TabulatedResponse(frequency_hz = frequency, response = matrix)

################################################################################
# Separate mechanical response from geometric transport
################################################################################

########################################
# Apply the matrix at the support port
########################################
support_without_cross_terms = drive.through(xt.mechanics.TabulatedResponse(
    frequency_hz = frequency, response = diagonal_matrix,
))
support_motion = drive.through(response)

########################################
# Transport the responded motion to the mounted magnet
########################################
magnet_motion = support_motion.transport(
    from_frames = [girder.nominal_frame], to_frames = [magnet.nominal_frame], target_names = ('magnet',),
)

########################################
# Bind the support and magnet as absolute outputs
########################################
mech.add_source('support_response', support_motion, targets = ['support'], reference = 'absolute')
mech.add_source('magnet_transport', magnet_motion, targets = ['magnet'], reference = 'absolute')

timings['matrix_response_and_transport_s'] = perf_counter() - started

################################################################################
# Verify the finite rigid transport
################################################################################

########################################
# Read the requested world poses
########################################
poses = mech.get_body_poses(time_s = INSPECT_TIME_S)

########################################
# Compare against direct frame composition
########################################
expected = poses['support'] @ girder.nominal_frame.inverse() @ magnet.nominal_frame
np.testing.assert_allclose(poses['magnet'].matrix, expected.matrix, atol = 2e-12, rtol = 0)

################################################################################
# Track the same launch before and after the response-driven magnet motion
################################################################################

########################################
# Launch on the design closed orbit and record one unperturbed traversal
########################################
started = perf_counter()
mech.apply(disabled_sources = tuple(mech.sources))
launch = line.twiss4d().particle_on_co.copy()
reference_particle = launch.copy()
line.track(reference_particle, num_turns = 1)

########################################
# Apply the full matrix and mount transform, then repeat the identical traversal
########################################
mech.apply(time_s = INSPECT_TIME_S)
moved_particle = launch.copy()
line.track(moved_particle, num_turns = 1)
beam_change_nm = np.array([
    float(moved_particle.x[0] - reference_particle.x[0]),
    float(moved_particle.y[0] - reference_particle.y[0]),
]) * 1e9
actual = line.survey(include_element_frames = True).get_frame(QUADRUPOLE, which = 'elem_start')
np.testing.assert_allclose(actual.matrix, poses['magnet'].matrix, rtol = 0, atol = 3e-12)
timings['Matched frozen-map beam tracking'] = perf_counter() - started

########################################
# Report the two distinct causes of magnet motion at one physical time
########################################
print('Mechanical cross coupling and rigid mounting geometry\n'
      '----------------------------------------------------')
print(f'Mount from support response port to magnet port x/y/s [m]: {MAGNET_MOUNT_M}')
print(f'Supplied yaw -> x coupling [m/rad]: {YAW_TO_X_M_PER_RAD:g}')
print(f'Supplied y -> pitch coupling [rad/m]: {Y_TO_PITCH_RAD_PER_M:g}')
print(f'Frozen comparison time [s]: {INSPECT_TIME_S:g}')
print(f'{"Response and observation port":<38} {"x [nm]":>12} {"y [nm]":>12} {"pitch [nrad]":>15}')
for label, record, port in [
    ('Support: diagonal response only', support_without_cross_terms, girder.nominal_frame),
    ('Support: mixed-coordinate response', support_motion, girder.nominal_frame),
    ('Magnet: response plus rigid transport', magnet_motion, magnet.nominal_frame),
]:
    values = record.evaluate(np.array([INSPECT_TIME_S]), np.array([port.XYZ]))[0, 0]
    print(f'{label:<38} {values[0] * 1e9:+12.5f} {values[1] * 1e9:+12.5f} {values[3] * 1e9:+15.5f}')
print('The first difference is supplied mechanical response at one port.')
print('The second difference is motion at another point on the same rigid body.')
print('For this aligned mount, dx_transport ~= s*yaw - y*roll and dy_transport ~= s*pitch + x*roll.')
print(f'Tracked one-turn change at the ring origin [nm]: dx = {beam_change_nm[0]:+.6f}, dy = {beam_change_nm[1]:+.6f}.')
print('Both traversals use the same design-closed-orbit launch; the second holds one mechanical state fixed.')
print('The frequency-domain response is periodic offline filtering; no causal startup or force law is inferred.')

################################################################################
# Draw the response port and the actual rigid mount lever
################################################################################

########################################
# Show the declared local port geometry in physical metres
########################################
setup_fig, setup_axes = plt.subplots(1, 2, figsize = (10, 3.8))
ax = setup_axes[0]
x_lever, height, longitudinal = MAGNET_MOUNT_M
quad_length = float(env[QUADRUPOLE].length)
ax.add_patch(plt.Rectangle((longitudinal, height - .15), quad_length, .30,
                           fc = 'tab:blue', ec = 'tab:blue', alpha = .2))
ax.plot(0., 0., 'kD', label = 'Support response port')
ax.plot(longitudinal, height, 'o', color = 'tab:blue', label = 'Magnet reference port')
ax.plot([0., longitudinal], [0., height], color = '0.35', lw = 1.5)
ax.plot([0., longitudinal, longitudinal], [0., 0., height], ':', color = '0.5')
ax.text(longitudinal / 2, -.07, f's = {longitudinal:g} m', ha = 'center', fontsize = 9)
ax.text(longitudinal + .03, height / 2, f'y = {height:g} m', va = 'center', fontsize = 9)
ax.text(.05, height + .15, f'Also x = {x_lever:g} m out of this view', fontsize = 9)
ax.set(xlabel = 'Support-local s [m]', ylabel = 'Support-local y [m]',
       title = 'One actual mount, declared independently of H(f)')
ax.set_aspect('equal', adjustable = 'box')
ax.margins(x = .2, y = .3)
port_handles, port_labels = ax.get_legend_handles_labels()
ax = setup_axes[1]
ax.set_axis_off()
for y, title, detail in [
    (.83, 'Six-coordinate support input', '[x, y, s, pitch, yaw, roll]'),
    (.50, 'Loaded response H(f), at the same port', 'Diagonal response + yaw -> x + y -> pitch'),
    (.17, 'Rigid transport to the magnet port', 'Use the declared lever once; preserve the phase'),
]:
    ax.text(.5, y, title + '\n' + detail, transform = ax.transAxes, ha = 'center', va = 'center', fontsize = 9,
            bbox = {'boxstyle': 'round,pad=.5', 'fc': '0.96', 'ec': '0.4'})
for y in (.69, .36):
    ax.annotate('', xy = (.5, y - .06), xytext = (.5, y + .04), xycoords = 'axes fraction',
                 arrowprops = {'arrowstyle': '->', 'color': '0.4'})
ax.set_title('Response and port transport are separate operations')
setup_fig.legend(port_handles, port_labels, loc = 'lower center', ncol = 2, fontsize = 8)
setup_fig.tight_layout(rect = (0., .11, 1., 1.))

################################################################################
# Compare diagonal response, mechanical cross terms and rigid lever-arm motion
################################################################################

########################################
# Use the same input record for each physical comparison
########################################
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (9, 7))
for record, label in [(support_without_cross_terms, 'Support: diagonal response'),
                       (support_motion, 'Support: cross terms enabled'),
                       (magnet_motion, 'Magnet: full response + mount transport')]:
    axes[0].plot(times, record.values[:, 0, 0] * 1e9, label = label)
for record, label in [(support_without_cross_terms, 'Diagonal response'), (support_motion, 'y-to-pitch coupling enabled')]:
    axes[1].plot(times, record.values[:, 0, 3] * 1e9, label = label)
axes[2].plot(times, (magnet_motion.values[:, 0, 1] - support_motion.values[:, 0, 1]) * 1e9,
             color = 'tab:green', label = 'Additional vertical displacement at the magnet port')
axes[0].set(ylabel = 'Local x [nm]', title = 'Same drive: distinguish response cross terms from a change of port')
axes[1].set(ylabel = 'Support pitch [nrad]')
axes[2].set(xlabel = 'Physical time [s]', ylabel = 'Mount-induced y [nm]', xlim = (0., 1.))
for ax in axes:
    ax.axvline(INSPECT_TIME_S, color = '0.4', ls = ':')
    ax.grid(alpha = .2)
    ax.margins(y = .3)
    ax.legend(fontsize = 8, loc = 'upper right')
fig.tight_layout()

########################################
# Keep an optional geometric animation available for interactive inspection
########################################
if MAKE_ANIMATION:
    animation = mech.animate(time_s = np.linspace(0, 1, 40), line = line,
        displacement_scale = 1e8, rotation_scale = 1e7, view = 'bodies')
mech.restore()

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
if PRINT_TIMINGS:
    print('\nReference CPU costs\n-------------------')
    for name, seconds in timings.items():
        print(f'{name:<40} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
