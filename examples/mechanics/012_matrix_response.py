# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Supply a complex 6x6 response at one rigid reference port.
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
MAGNET_MOUNT_M = (.1, 0., 1.)         # Lever arm, handled separately by transport

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

################################################################################
# Set up the ordinary Xtrack line
################################################################################

########################################
# Start performance monitoring
########################################
timings = {}

########################################
# Time line load and copy
########################################
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

########################################
# Time reference adoption
########################################
started = perf_counter()

########################################
# Adopt existing offsets as the deliberate reference
########################################
line.mechanics.adopt_reference()

timings['reference_adoption_s'] = perf_counter() - started

########################################
# Time geometry selection
########################################
started = perf_counter()

########################################
# Survey the magnet's nominal reference port
########################################
nominal = line.survey(include_element_frames = True)

timings['geometry_selection_s'] = perf_counter() - started
########################################
# Time mechanics definition
########################################
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

########################################
# Start timing response preparation
########################################
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

########################################
# Record matrix-response and transport cost
########################################
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

########################################
# Compare all coordinate amplitudes at the input and output ports
########################################
print('Six-coordinate response and rigid port transport\n-------------------------------------------------')
print('Coordinate         Drive RMS       Support RMS       Magnet RMS')
for index, name in enumerate(('x [nm]', 'y [nm]', 's [nm]', 'pitch [nrad]', 'yaw [nrad]', 'roll [nrad]')):
    values = [np.std(record.values[:, 0, index]) * 1e9 for record in (drive, support_motion, magnet_motion)]
    print(f'{name:<15} {values[0]:13.6f} {values[1]:17.6f} {values[2]:16.6f}')
print(f'Yaw → x response coupling [m/rad]: {YAW_TO_X_M_PER_RAD:g}')
print(f'y → pitch response coupling [rad/m]: {Y_TO_PITCH_RAD_PER_M:g}')
print(f'Mount lever arm x/y/s [m]: {MAGNET_MOUNT_M}')
print('Matrix coupling is a supplied small-signal response. Port transport is exact rigid geometry.')

################################################################################
# Inspect coordinate coupling and geometry
################################################################################

########################################
# Prepare translation and pitch history panels
########################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))

########################################
# Plot the drive, support and transported magnet
########################################
for record, label, style in [(drive, 'Drive', '--'), (support_motion, 'Support', '-'), (magnet_motion, 'Magnet', ':')]:
    axes[0].plot(times, record.values[:, 0, 0] * 1e9, style, label = label)
    axes[1].plot(times, record.values[:, 0, 3] * 1e9, style, label = label)

########################################
# Label the diagnostic histories
########################################
axes[0].set(ylabel = 'Local x [nm]', title = 'Matrix coupling plus independent mount transport')
axes[1].set(xlabel = 'Physical time [s]', ylabel = 'Survey pitch [nrad]', xlim = (0, 1))
axes[0].legend()
axes[1].legend()
fig.tight_layout()

########################################
# Draw the magnified local mounting geometry
########################################
layout_fig, layout_axes = mech.plot(
    time_s = INSPECT_TIME_S, line = line, displacement_scale = 1e8, rotation_scale = 1e7, view = 'bodies',
)

########################################
# Optionally animate translation, orientation and geometry
########################################
if MAKE_ANIMATION:
    animation = mech.animate(time_s = np.linspace(0, 1, 40), line = line,
        displacement_scale = 1e8, rotation_scale = 1e7, view = 'bodies')

################################################################################
# Review execution costs and figures
################################################################################

########################################
# Print the measured costs
########################################
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<40} {seconds:9.4f} s')

########################################
# Display the figures
########################################
plt.show()
