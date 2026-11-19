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
# Six-coordinate drive and response
########################################
LATTICE_FILE = 'test_data/example_lattices/demo_ring/lattice.py'
QUADRUPOLE = 'arc.s1.c1.qf.in'
TRANSLATION_PEAK_M = (10e-9, 50e-9, 5e-9)
ANGLE_PEAK_RAD = (2e-9, 3e-9, 4e-9)      # Survey pitch, yaw, roll
DRIVE_HZ = (2., 5., 3., 5., 7., 11.)
RESONANCE_HZ = (12., 10., 25., 15., 18., 22.)
DAMPING = .15
YAW_TO_X_M_PER_RAD = 1.5                 # Supplied response coupling, same port
Y_TO_PITCH_RAD_PER_M = .2
MAGNET_MOUNT_M = (.1, 0., 1.)            # Lever arm, handled separately by transport

########################################
# Record and display
########################################
N_SAMPLES     = 2048
SAMPLE_STEP_S = 1 / 256
INSPECT_TIME_S = .125
MAKE_ANIMATION = False

################################################################################
# Build a support and a magnet with an explicit local mount
################################################################################
started = perf_counter()
repo_root = Path(__file__).resolve().parents[2]
nominal_env = xt.load(repo_root / LATTICE_FILE)
line = nominal_env['demo'].copy()
env = line.env
env['demo'] = line
mech = env.mechanics
line.mechanics.adopt_reference()
nominal = line.survey(include_element_frames = True)
magnet_frame = nominal.get_frame(QUADRUPOLE, which = 'elem_start')
mount = xt.Frame.from_survey_angles(X = MAGNET_MOUNT_M[0], Y = MAGNET_MOUNT_M[1], Z = MAGNET_MOUNT_M[2])
girder = mech.new_girder('support', anchor = magnet_frame @ mount.inverse())
magnet = mech.new_body('magnet', parent = girder, mount = mount, elements = [QUADRUPOLE], line = line)
timings = {'setup_s': perf_counter() - started}

################################################################################
# Apply a user-supplied matrix with explicit mixed SI units
################################################################################
started = perf_counter()
times = np.arange(N_SAMPLES) * SAMPLE_STEP_S
amplitude = np.array([*TRANSLATION_PEAK_M, *ANGLE_PEAK_RAD])
drive_values = amplitude[None, None, :] * np.sin(2 * np.pi * times[:, None, None] * np.array(DRIVE_HZ))
drive = xt.mechanics.RigidMotion(times, drive_values, ('support',))
frequency = np.fft.rfftfreq(N_SAMPLES, SAMPLE_STEP_S)
matrix = np.zeros((len(frequency), 6, 6), dtype = complex)
for index, resonance in enumerate(RESONANCE_HZ):
    matrix[:, index, index] = xt.mechanics.BaseExcitedOscillator(resonance, DAMPING).evaluate(frequency)
matrix[:, 0, 4] = YAW_TO_X_M_PER_RAD * matrix[:, 0, 0]
matrix[:, 3, 1] = Y_TO_PITCH_RAD_PER_M * matrix[:, 3, 3]
response = xt.mechanics.TabulatedResponse(frequency_hz = frequency, response = matrix)
support_motion = drive.through(response)
magnet_motion = support_motion.transport(
    from_frames = [girder.nominal_frame], to_frames = [magnet.nominal_frame], target_names = ('magnet',),
)
mech.add_source('support_response', support_motion, targets = ['support'], reference = 'absolute')
mech.add_source('magnet_transport', magnet_motion, targets = ['magnet'], reference = 'absolute')
timings['matrix_response_and_transport_s'] = perf_counter() - started

########################################
# Verify that the transported magnet follows the same rigid transform
########################################
poses = mech.get_body_poses(time_s = INSPECT_TIME_S)
expected = poses['support'] @ girder.nominal_frame.inverse() @ magnet.nominal_frame
np.testing.assert_allclose(poses['magnet'].matrix, expected.matrix, atol = 2e-12, rtol = 0)
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
# Inspect mixed-coordinate histories and magnified orientation
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
for record, label, style in [(drive, 'Drive', '--'), (support_motion, 'Support', '-'), (magnet_motion, 'Magnet', ':')]:
    axes[0].plot(times, record.values[:, 0, 0] * 1e9, style, label = label)
    axes[1].plot(times, record.values[:, 0, 3] * 1e9, style, label = label)
axes[0].set(ylabel = 'Local x [nm]', title = 'Matrix coupling plus independent mount transport')
axes[1].set(xlabel = 'Physical time [s]', ylabel = 'Survey pitch [nrad]', xlim = (0, 1))
axes[0].legend()
axes[1].legend()
fig.tight_layout()
layout_fig, layout_axes = mech.plot(
    time_s = INSPECT_TIME_S, line = line, displacement_scale = 1e8, rotation_scale = 1e7, view = 'bodies',
)
if MAKE_ANIMATION:
    animation = mech.animate(time_s = np.linspace(0, 1, 40), line = line,
        displacement_scale = 1e8, rotation_scale = 1e7, view = 'bodies')
print('\nReference CPU costs\n-------------------')
for name, seconds in timings.items():
    print(f'{name:<40} {seconds:9.4f} s')
plt.show()
